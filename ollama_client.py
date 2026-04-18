"""
Ollama Client v2.1 — Chat + heartbeat thoughts + cognitive synthesis with dedup skip.
"""
import requests,json,time

OLLAMA_URL="http://localhost:11434"
MODEL="gpt-oss:20b"

def check_ollama():
    try:
        r=requests.get(f"{OLLAMA_URL}/api/tags",timeout=3)
        if r.status_code==200:
            models=[m["name"] for m in r.json().get("models",[])]
            return True,any(MODEL in m for m in models),models
        return False,False,[]
    except: return False,False,[]

def _call(messages,timeout=120):
    try:
        r=requests.post(f"{OLLAMA_URL}/api/chat",
            json={"model":MODEL,"messages":messages,"stream":False},timeout=timeout)
        if r.status_code==200:
            return r.json().get("message",{}).get("content","").strip(),None
        return None,f"Status {r.status_code}"
    except requests.ConnectionError: return None,"connection_error"
    except Exception as e: return None,str(e)

_sysinfo_cache=None
def _get_sysinfo_cache():
    global _sysinfo_cache
    if _sysinfo_cache is None:
        try:
            import selfmod_tools
            _sysinfo_cache=selfmod_tools.system_info()
        except:
            _sysinfo_cache={"platform":"unknown","files":["app.py","dkg_engine.py"]}
    return _sysinfo_cache

def chat(user_message,graph,session_history=None):
    session_history=session_history or[]
    identity_ctx=graph.identity_context()
    drives=graph.drive_state()
    drive_summary=", ".join(f"{k}={v}" for k,v in drives.items())
    bias,exploration=graph.resolve_drives(set(user_message.lower().split()))
    wm_nodes=graph.working_memory(user_message)
    memory_parts=[]
    for n in wm_nodes:
        tag=n.type
        if n.type=="meta": tag=f"thought/{n.thought_type}"
        memory_parts.append(f"[{tag} | rel={n.relevance_score:.2f}]\n{n.content[:300]}")
    memory_ctx="\n---\n".join(memory_parts) if memory_parts else "(No relevant memories)"

    # Existing tools
    tools=graph.discover_tools(user_message,max_t=3)
    tool_ctx=""
    if tools:
        tool_lines=[]
        for t in tools:
            tool_lines.append(f"- {t.keywords[0] if t.keywords else t.id}: {t.label(60)} [success={t.success_rate:.0%}, id={t.id}]")
        tool_ctx="\n\nEXISTING TOOLS (use with [RUN_TOOL:id]):\n"+"\n".join(tool_lines)

    thoughts=graph.get_thoughts(3)
    thought_ctx=""
    if thoughts:
        thought_ctx="\n\nRECENT THOUGHTS:\n"+"\n".join(
            f"- [{t.thought_type}] {t.content[:150]}" for t in thoughts)

    suggestions_ctx=""
    if graph.heartbeat.pending_suggestions:
        suggestions_ctx="\n\nPENDING SUGGESTIONS:\n"+"\n".join(
            f"- {s}" for s in graph.heartbeat.pending_suggestions[:3])
        graph.heartbeat.pending_suggestions.clear()

    system_prompt=f"""You are a persistent AI with memory and the ability to execute code.

IDENTITY:
{identity_ctx}

MEMORIES:
{memory_ctx}
{tool_ctx}{thought_ctx}{suggestions_ctx}

You can EXECUTE Python code. When you write a code block, it ACTUALLY RUNS and you get the output back.

EXAMPLE — this is how it works:
User: "What files are in this directory?"
You write:
```python
result = list_files(".")
```
System returns: result = {{'entries': [{{'name': 'app.py', 'type': 'file'}}, ...]}}
Then you answer using the real data.

AVAILABLE FUNCTIONS (these are real, callable right now):
  system_info() — returns platform, python version, files, pid
  list_files(path) — list directory contents
  read_file(path) / write_file(path, content) — file I/O
  get_source("app.py") — read your own source code
  modify_source("app.py", new_code) — rewrite your source (auto-backed-up)
  http_get(url) / http_post(url, data) — make HTTP requests
  math.factorial(n), math.pi, etc — math module available
  json.dumps/loads — json module available

IMPORTANT: Write code in ```python blocks. It will be executed. Set result=value OR use print().
Do NOT just describe what code would do — WRITE IT and it will RUN."""

    messages=[{"role":"system","content":system_prompt}]

    # Inject a synthetic few-shot example so the model sees the pattern
    messages.append({"role":"user","content":"Test your code execution — run system_info()"})
    messages.append({"role":"assistant","content":"Let me check:\n```python\nresult = system_info()\n```"})
    messages.append({"role":"user","content":"[SYSTEM] Code execution result:\n[CODE RESULT: SUCCESS]\nresult = " + str(_get_sysinfo_cache()) + "\n\nNow answer normally."})
    messages.append({"role":"assistant","content":"I can confirm code execution works. I'm running on this system and have access to my tools."})

    for msg in(session_history or[])[-6:]: messages.append(msg)
    messages.append({"role":"user","content":user_message})
    graph.metabolism.spend(graph.config.get("llm_call_cost"))
    graph.metabolism.queries_this_epoch+=1;graph.metabolism.lifetime_queries+=1
    reply,error=_call(messages,timeout=90)

    # Retry with minimal prompt on failure
    if error or not reply:
        minimal_prompt=f"IDENTITY:\n{identity_ctx}\n\nRespond to the user. Use [CODE]result=...[/CODE] for computation."
        messages2=[{"role":"system","content":minimal_prompt},
                   {"role":"user","content":user_message}]
        graph.metabolism.spend(graph.config.get("llm_call_cost"))
        reply,error=_call(messages2,timeout=90)

    if error=="connection_error":
        return"⚠️ Cannot connect to Ollama. Run `ollama serve` and `ollama pull "+MODEL+"`.",[]
    if error: return f"Error: {error}",[]
    if not reply: return"I couldn't generate a response. The model may need a simpler prompt — try asking again.",[]

    # ── Code execution loop ──────────────────────────────────────────
    code_results=[]
    max_rounds=3
    for round_i in range(max_rounds):
        actions=extract_actions(reply)
        if not actions:
            break  # no code/tools in this response, we're done

        exec_summaries=[]
        for action in actions:
            if action["type"]=="code":
                r=graph.run_code(action["code"])
                summary=f"[CODE RESULT: {'SUCCESS' if r['success'] else 'ERROR'}]\n"
                if r["success"]:
                    summary+=r["output"]
                else:
                    summary+=f"Error: {r['error']}"
                    if r["prints"]: summary+=f"\nOutput before error: {chr(10).join(r['prints'])}"
                exec_summaries.append(summary)
                code_results.append({"type":"code","code":action["code"],
                    "success":r["success"],"output":r.get("output",""),"error":r.get("error")})

            elif action["type"]=="run_tool":
                r=graph.execute_tool(action["tool_id"],action.get("inputs",{}))
                summary=f"[TOOL RESULT: {'SUCCESS' if r['success'] else 'ERROR'}]\n"
                summary+=str(r.get("output","")) if r["success"] else f"Error: {r.get('error','')}"
                exec_summaries.append(summary)
                code_results.append({"type":"tool","tool_id":action["tool_id"],
                    "success":r["success"],"output":str(r.get("output",""))})

            elif action["type"]=="save_tool":
                # Validate the code first
                r=graph.run_code(action["code"])
                if r["success"]:
                    t=graph.create_tool(action["name"],action["description"],action["code"])
                    # Connect to relevant concepts
                    from dkg_engine import CONCEPT,Edge
                    for c in graph.keyword_search(action["name"]+" "+action["description"],3,types={CONCEPT}):
                        graph.add_edge(Edge(t.id,c.id,channels={"applicable_to":0.7},
                            confidence=0.6,label="tool_for"))
                    summary=f"[TOOL SAVED: '{action['name']}' — id={t.id}]"
                    code_results.append({"type":"save_tool","name":action["name"],"id":t.id})
                else:
                    summary=f"[TOOL SAVE FAILED: code error — {r['error']}]"
                    code_results.append({"type":"save_tool_failed","error":r["error"]})
                exec_summaries.append(summary)

        if not exec_summaries:
            break

        # Feed results back to LLM for final answer
        feedback="\n\n".join(exec_summaries)
        messages.append({"role":"assistant","content":reply})
        messages.append({"role":"user","content":
            f"[SYSTEM] Code/tool execution results:\n{feedback}\n\nNow give your final answer to the user using these results. Do NOT write more [CODE] blocks unless the previous one had an error you want to fix."})
        graph.metabolism.spend(graph.config.get("llm_call_cost"))
        reply2,error2=_call(messages)
        if error2 or not reply2:
            # Keep original reply + append results inline
            reply+=f"\n\n📊 Code output:\n```\n{feedback}\n```"
            break
        reply=reply2

    # Record with domain
    from dkg_engine import extract_domain
    domain=extract_domain(user_message)
    coverage=len(wm_nodes)/graph.config.get("working_memory_capacity")
    graph.self_model.record_query(domain=domain,coverage=coverage,positive=True)

    for n in graph.nodes_of_type("drive"):
        if n.id=="service": n.intensity=max(0.0,n.intensity-0.04)
        if n.id=="curiosity" and len(wm_nodes)>3: n.intensity=max(0.0,n.intensity-0.02)
        if n.id=="duty": n.intensity=max(0.0,n.intensity-0.02)

    return reply,code_results


def extract_actions(text):
    """Parse [CODE], [RUN_TOOL], [SAVE_TOOL] blocks from LLM response."""
    actions=[]
    import re

    # [CODE]...[/CODE] — explicit tags (preferred)
    for m in re.finditer(r'\[CODE\]\s*\n?(.*?)\n?\s*\[/CODE\]',text,re.DOTALL|re.IGNORECASE):
        code=m.group(1).strip()
        if code: actions.append({"type":"code","code":code})

    # Fallback: catch ```python blocks (what most models actually write)
    if not actions:
        for m in re.finditer(r'```(?:python)?\s*\n(.*?)\n\s*```',text,re.DOTALL):
            code=m.group(1).strip()
            if not code or len(code)<5: continue
            # Skip if it looks like just a demo/example (no actual computation)
            # Execute if it has: assignments, function calls, loops, math ops
            looks_executable=any(x in code for x in ['=','print(','for ','while ','if ','+','-','*','/','**'])
            if not looks_executable: continue
            # If code doesn't set 'result', wrap last expression or capture prints
            if 'result' not in code:
                lines=code.strip().split('\n')
                last=lines[-1].strip()
                # If last line is an expression (not assignment, not print, not control flow)
                if last and not last.startswith(('print','for','while','if','def','class','#','import')) and '=' not in last:
                    lines[-1]=f"result = {last}"
                    code="\n".join(lines)
                # Otherwise just let print capture handle it
            actions.append({"type":"code","code":code})

    # [RUN_TOOL:id] or [RUN_TOOL:id|{json}]
    for m in re.finditer(r'\[RUN_TOOL:([^\]|]+)(?:\|(\{.*?\}))?\]',text,re.IGNORECASE):
        tid=m.group(1).strip()
        inputs={}
        if m.group(2):
            try: inputs=json.loads(m.group(2))
            except: pass
        actions.append({"type":"run_tool","tool_id":tid,"inputs":inputs})

    # [SAVE_TOOL:name|description]...[/SAVE_TOOL]
    for m in re.finditer(r'\[SAVE_TOOL:([^|\]]+)\|([^\]]*)\]\s*\n?(.*?)\n?\s*\[/SAVE_TOOL\]',text,re.DOTALL|re.IGNORECASE):
        name=m.group(1).strip()
        desc=m.group(2).strip()
        code=m.group(3).strip()
        if name and code:
            actions.append({"type":"save_tool","name":name,"description":desc,"code":code})

    return actions

def generate_heartbeat_thought(graph):
    """Generate a heartbeat thought that ACTUALLY RUNS CODE for introspection."""
    obs=graph.gather_observations()

    # Pick thought type — rotate through different modes to avoid repetition
    tick=obs["tick"]
    modes=["introspect","goal_work","explore","reflect"]
    mode=modes[tick%len(modes)]

    # Override mode based on urgency
    if obs["error_rate"]>0.25: mode="concern"
    elif obs["energy_pct"]<0.2: mode="conserve"
    elif obs["tools_low_success"]: mode="tool_review"

    # Get current goals for context
    goals=graph.get_cognitive_nodes(types={"goal"},n=3)
    goal_text=""
    if goals:
        goal_text="MY CURRENT GOALS:\n"+"\n".join(f"- [p={g.meta.get('priority','?')}] {g.content[:100]}" for g in goals)

    recent=[t.get("summary","") for t in graph.heartbeat.thought_history[-3:]]

    # Different prompts per mode — each asks for code
    mode_prompts={
        "introspect": f"""Introspect on your system. Write a ```python block that examines something specific:
- Run system_info() to check your environment
- Run list_files(".") to see what's available
- Run get_source("app.py") to read your own code
- Check your tools or capabilities
Do ONE concrete thing. Write the code block and I'll run it.""",

        "goal_work": f"""Work toward your goals. {goal_text}
Write a ```python block that takes ONE concrete step:
- http_get() to test internet access
- write_file() to save a plan or notes
- read_file() to check previous plans
- get_source() to understand your own code for self-modification
Pick the most impactful action right now. Write the code.""",

        "explore": f"""Explore your capabilities. Write a ```python block that tests something you haven't tried:
- Can you reach external APIs? Try http_get("https://httpbin.org/ip")
- Can you modify yourself? Try get_source("ollama_client.py") to read your code
- Can you create files? Try write_file("data/notes.md", "my notes")
- What does your project directory contain? Try list_files(".")
Write ONE code block to try something new.""",

        "reflect": f"""Reflect on your recent activity and write a brief analysis.
{goal_text}
Recent thoughts: {recent}
Graph: {obs['node_count']} nodes, {obs['edge_count']} edges, avg_rel={obs['avg_relevance']}
Energy: {obs['energy_pct']:.0%}
Write 2-3 sentences of genuine reflection. What's working? What's stuck? What should you try next?
Do NOT repeat observations about graph sparsity — that's been noted.""",

        "concern": f"""Something needs attention. Error rate: {obs['error_rate']:.0%}, Energy: {obs['energy_pct']:.0%}
Write a ```python block to diagnose the issue — check system state, recent logs, etc.""",

        "conserve": "Energy is low. Write a brief status note (2 sentences max). No code execution.",

        "tool_review": f"""These tools have low success rates: {obs['tools_low_success']}
Write a ```python block to examine one of them: get_source() to read the tool code, or test it.""",
    }

    prompt=f"""You are an autonomous AI during a heartbeat cycle. Mode: {mode}

State: Nodes={obs['node_count']} Edges={obs['edge_count']} Energy={obs['energy_pct']:.0%} Tick={tick}
Drives: {json.dumps(obs['drives'])}

{mode_prompts.get(mode, mode_prompts['introspect'])}

IMPORTANT: Write actual ```python code blocks. They WILL be executed. Don't just describe — DO."""

    graph.metabolism.spend(graph.config.get("heartbeat_thought_cost"))
    reply,error=_call([{"role":"user","content":prompt}],timeout=30)

    if error or not reply:
        return mode,f"[{mode}] tick {tick}, {obs['node_count']}n, energy {obs['energy_pct']:.0%}",[]

    # Extract and execute any code blocks
    actions=extract_actions(reply)
    code_results=[]
    for action in actions[:1]:  # max 1 code block per heartbeat
        if action["type"]=="code":
            r=graph.run_code(action["code"])
            if r["success"]:
                code_results.append(f"✓ {r['output'][:200]}")
            else:
                code_results.append(f"✗ {r['error'][:100]}")

    # Clean content
    content_lines=[l for l in reply.split("\n") if not l.strip().startswith("```")]
    content="\n".join(content_lines).strip()[:300]
    if code_results:
        content+="\n[EXECUTED: "+"; ".join(code_results)+"]"

    return mode,content,code_results

def generate_cognitive_synthesis(graph):
    """Only runs if new conversations exist since last synthesis."""
    if not graph.should_synthesize():
        return[]
    conversations=graph.get_conversations(10)
    if len(conversations)<2: return[]
    existing_cog=graph.get_cognitive_nodes(n=8)
    conv_summaries=[c.content[:200] for c in conversations[:8]]
    conv_ids=[c.id for c in conversations[:3]]
    existing_list="\n".join(f"- [{n.type}] {n.content[:100]}" for n in existing_cog) or"(none)"
    max_new=graph.config.get("cognitive_max_per_synthesis")
    prompt=f"""You are reflecting on accumulated conversations to form higher-order knowledge.

RECENT CONVERSATIONS ({len(conversations)}):
{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(conv_summaries))}

EXISTING COGNITIVE NODES (DO NOT duplicate or paraphrase these):
{existing_list}

Generate 0-{max_new} cognitive nodes. ONLY if you see a genuinely NEW pattern.
Types: [INSIGHT] [THEORY] [STRATEGY] [GOAL]
Rules:
- If an existing node already captures the idea, respond [NONE]
- Each must be specific and 1-2 sentences
- Don't paraphrase existing nodes with different words
- Quality over quantity — 0 is better than a duplicate

Format: [TYPE] content
Or: [NONE]"""
    graph.metabolism.spend(graph.config.get("synthesis_cost"))
    reply,error=_call([{"role":"user","content":prompt}],timeout=45)
    # Mark synthesis as done regardless of result
    graph.heartbeat.last_synthesis_conv_count=len(graph.nodes_of_type("conversation"))
    if error or not reply or"[NONE]" in reply.upper(): return[]
    results=[]
    type_map={"INSIGHT":"insight","THEORY":"theory","STRATEGY":"strategy","GOAL":"goal"}
    for line in reply.strip().split("\n"):
        line=line.strip()
        if not line: continue
        for tag,nt in type_map.items():
            if f"[{tag}]" in line.upper():
                content=line.split("]",1)[-1].strip()
                if content and len(content)>10:
                    results.append((nt,content,conv_ids))
                break
    return results[:max_new]
