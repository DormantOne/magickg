"""
DKG Go v13 — Brain (White, full pipeline) vs Configurable Opponent (Black)
  + Heat Map perceptual layer + Fresh-eyes Critic + Working Strategies.
Heat map computes per-point impact (capture/save/reduce/connect/territory/structure).
Heat analysis produces a position read (LLM) that's stored as HEAT_READ in KG.
Move prompt receives compact heat summary instead of raw grid.
"""
import os,time,threading,json,re,traceback,random,shutil,requests
from flask import Flask,render_template,request,jsonify
from dkg_engine import DKG,Node,Edge,Config,INSIGHT,THEORY,STRATEGY,GOAL,TOOL,CONVERSATION,IDENTITY,DRIVE,META,WORKING_STRATEGY,HEAT_READ
from go_engine import GoGame,BLACK,WHITE,EMPTY,opponent
import ollama_client

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — load from config.json, fall back to defaults
# ═════════════════════════════════════════════════════════════════════════════
CFG_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.json")
def load_config():
    defaults={
        "timeout_tool_select":300,"timeout_knowledge_select":300,"timeout_move":300,
        "timeout_distill":300,"timeout_review":300,
        "timeout_tool_retry":300,"timeout_correction":300,"timeout_lens_select":300,
        "timeout_bare_llm":120,"timeout_critic":180,
        "board_size":9,"komi":5.5,"move_cap":120,"min_moves_before_pass":60,
        "auto_speed":4,"synthesis_interval":8,
        "opponent_type":"bare_llm",
        "enable_critic":True,"critic_dream_sample":3,
        "enable_working_strategy":True,
        "enable_heat_analysis":True,"enable_heat_kg":True,
        "heat_anchors_k":5,"heat_suppression":2,
        "timeout_heat_analysis":90,
        "enable_tool_select":True,"enable_knowledge_select":True,
        "enable_post_game_review":True,
        "enable_lens_select":True,"enable_tool_eval":True,"enable_knowledge_decay":True,
        "max_tools_per_move":3,"max_knowledge_nodes":3,
        "lenses":["defense","attack","territory","endgame","tactics"],
        "funnel_1pass_max":20,"funnel_2pass_max":50,
        "tool_retire_after_n_uses":20,"tool_retire_min_delta":-1.0,
        "knowledge_win_boost":0.08,"knowledge_lose_penalty":0.05,"knowledge_unused_decay":0.02,
        "probe_max_time":3600,"port":5028,
    }
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH) as f: raw=json.load(f)
            for k,v in raw.items():
                if not k.startswith("//") and k in defaults:
                    defaults[k]=v
            print(f"  Config: loaded {CFG_PATH}")
        except Exception as e: print(f"  Config: error {e}, using defaults")
    else: print("  Config: no config.json, using defaults")
    return defaults

CFG=load_config()

app=Flask(__name__)
game=GoGame(size=CFG["board_size"],komi=CFG["komi"],move_cap=CFG["move_cap"])
players={}
game_log=[]
auto_play=False
auto_speed=CFG["auto_speed"]
game_lock=threading.Lock()

SAVE_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
SNAP_DIR=os.path.join(SAVE_DIR,"snapshots")
RESULTS_DIR=os.path.join(SAVE_DIR,"results")
GAMES_DIR=os.path.join(SAVE_DIR,"games")
ROUNDS_DIR=os.path.join(SAVE_DIR,"rounds")
for d in[SAVE_DIR,SNAP_DIR,RESULTS_DIR,GAMES_DIR,ROUNDS_DIR]: os.makedirs(d,exist_ok=True)

# ── Persistent scoreboard ────────────────────────────────────────────────────
SCORE_PATH=os.path.join(SAVE_DIR,"scoreboard.json")
def _load_sb():
    if os.path.exists(SCORE_PATH):
        with open(SCORE_PATH) as f: return json.load(f)
    return {"black_wins":0,"white_wins":0,"draws":0,"games":[],"total_games":0}
def _save_sb(sb):
    with open(SCORE_PATH,"w") as f: json.dump(sb,f,indent=1)
scoreboard=_load_sb()

EVAL_PATH=os.path.join(SAVE_DIR,"eval_scoreboard.json")
def _load_eval_sb():
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f: return json.load(f)
    return{"games":[]}
def _save_eval_sb(sb):
    with open(EVAL_PATH,"w") as f: json.dump(sb,f,indent=1)
eval_scoreboard=_load_eval_sb()

# ═════════════════════════════════════════════════════════════════════════════
#  ROUND STATE
# ═════════════════════════════════════════════════════════════════════════════
ROUND_PATH=os.path.join(SAVE_DIR,"round_history.json")
def _load_round_history():
    if os.path.exists(ROUND_PATH):
        with open(ROUND_PATH) as f: return json.load(f)
    return[]
def _save_round_history(h):
    with open(ROUND_PATH,"w") as f: json.dump(h,f,indent=1)
round_history=_load_round_history()

round_state={
    "round":len(round_history)+1,"phase":"train","phase_num":1,
    "phase_label":"TRAIN","current_round_results":None,
}

# ── Per-game metrics ──────────────────────────────────────────────────────────
def _new_game_metrics():
    return {"timeouts":0,"timeout_details":[],"parse_failures":0,"retries":0,
            "fallbacks":0,"latencies":[],"parse_methods":{},"knowledge_used":[],
            "confidence_scores":[],"total_duration":0,"moves_log":[],"_start_time":time.time(),
            "pipeline_steps":[]}
game_metrics=_new_game_metrics()

# ── Pre-flight ────────────────────────────────────────────────────────────────
def preflight_check():
    print("  ── Pre-flight ──")
    try:
        t0=time.time()
        r=requests.get("http://localhost:11434/api/tags",timeout=5)
        latency=round(time.time()-t0,2)
        if r.status_code==200:
            models=[m["name"] for m in r.json().get("models",[])]
            has_model=any(ollama_client.MODEL in m for m in models)
            print(f"  Ollama: ✓ {latency}s | Model: {'✓' if has_model else '✗'} {ollama_client.MODEL}")
        else: print(f"  Ollama: ✗ status {r.status_code}")
    except Exception as e: print(f"  Ollama: ✗ {e}")
    print(f"  Timeouts: move={CFG['timeout_move']}s tool={CFG['timeout_tool_select']}s knowledge={CFG['timeout_knowledge_select']}s")
    print("  ── Ready ──")

# ═════════════════════════════════════════════════════════════════════════════
#  HEURISTIC / RANDOM PLAYERS
# ═════════════════════════════════════════════════════════════════════════════
class HeuristicPlayer:
    """Heuristic opponent with influence awareness, territory sense, and tenuki.
    Priority: capture > save group > reduce liberties > influence-based expansion.
    Deterministic but strategically aware — not trivially exploitable."""

    def choose_move(self,g,color):
        legal=[m for m in g.legal_moves(color) if m[0]>=0]
        if not legal: return(-1,-1),"pass","heuristic:no_moves"
        opp=opponent(color);sz=g.size;center=sz//2

        # ── 1. Capture: take groups in atari ──
        for r,c in legal:
            for nr,nc in g.neighbors(r,c):
                if g.board[nr][nc]==opp:
                    grp,libs=g.group_and_liberties(nr,nc)
                    if len(libs)==1 and(r,c) in libs:
                        return(r,c),f"cap ({r},{c})","heuristic:capture"

        # ── 2. Save: defend own groups in atari ──
        for r,c in legal:
            old=g.copy_board();g.board[r][c]=color;saved=False
            for nr,nc in g.neighbors(r,c):
                if g.board[nr][nc]==color:
                    _,libs=g.group_and_liberties(nr,nc);g.board[r][c]=EMPTY
                    _,libs2=g.group_and_liberties(nr,nc);g.board[r][c]=color
                    if len(libs2)<=1 and len(libs)>1: saved=True;break
            g.board=old
            if saved: return(r,c),f"def ({r},{c})","heuristic:defend"

        # ── 3. Avoid self-atari: filter out moves that put own stone in atari ──
        safe_legal=[]
        for r,c in legal:
            old=g.copy_board();g.board[r][c]=color
            _,libs=g.group_and_liberties(r,c)
            g.board=old
            if len(libs)>=2: safe_legal.append((r,c))
        if not safe_legal: safe_legal=legal  # all moves are bad, use them anyway

        # ── 4. Reduce: put opponent groups into atari ──
        for r,c in safe_legal:
            for nr,nc in g.neighbors(r,c):
                if g.board[nr][nc]==opp:
                    grp,libs=g.group_and_liberties(nr,nc)
                    if len(libs)==2 and(r,c) in libs:
                        return(r,c),f"red ({r},{c})","heuristic:reduce"

        # ── 5. Score all safe moves by influence + territory + structure ──
        scores={}
        # Precompute influence map
        influence=[[0.0]*sz for _ in range(sz)]
        for ir in range(sz):
            for ic in range(sz):
                if g.board[ir][ic]==EMPTY: continue
                sign=1.0 if g.board[ir][ic]==color else -1.0
                for jr in range(sz):
                    for jc in range(sz):
                        dist=abs(ir-jr)+abs(ic-jc)
                        if dist<=4: influence[jr][jc]+=sign*max(0,1.0-dist*0.22)

        for r,c in safe_legal:
            s=0.0
            # Influence: prefer moves in areas we already influence
            s+=influence[r][c]*0.8
            # Territory: edges and corners are valuable for territory
            edge_r=min(r,sz-1-r);edge_c=min(c,sz-1-c)
            edge_dist=min(edge_r,edge_c)
            if edge_dist<=2: s+=0.5  # near edge = territory potential
            # Star points and key intersections (9x9: 2,2 / 2,6 / 6,2 / 6,6 / 4,4)
            star=[(2,2),(2,sz-3),(sz-3,2),(sz-3,sz-3),(center,center)]
            if(r,c) in star and g.board[r][c]==EMPTY: s+=1.2
            # Connectivity: prefer moves adjacent to own stones
            own_neighbors=sum(1 for nr,nc in g.neighbors(r,c) if g.board[nr][nc]==color)
            opp_neighbors=sum(1 for nr,nc in g.neighbors(r,c) if g.board[nr][nc]==opp)
            s+=own_neighbors*0.4  # build connected groups
            s+=opp_neighbors*0.2  # contest opponent territory
            # Avoid playing directly on edge row/col unless it's a star point
            if(r==0 or r==sz-1 or c==0 or c==sz-1) and (r,c) not in star: s-=0.6
            # Center pull (mild)
            center_dist=abs(r-center)+abs(c-center)
            s+=max(0,1.0-center_dist*0.1)
            # Tenuki bonus: if no own stones within distance 3, slight bonus (spread out)
            nearest_own=min((abs(r-ir)+abs(c-ic) for ir in range(sz) for ic in range(sz) if g.board[ir][ic]==color),default=99)
            if nearest_own>=3: s+=0.3
            scores[(r,c)]=s

        # Pick best scoring move
        best=max(safe_legal,key=lambda p:scores.get(p,0))
        return best,f"inf ({best[0]},{best[1]}) s={scores[best]:.1f}","heuristic:influence"

class RandomPlayer:
    def choose_move(self,g,color):
        legal=[m for m in g.legal_moves(color) if m[0]>=0]
        if not legal: return(-1,-1),"pass","random:none"
        m=random.choice(legal);return m,f"rnd {m}","random"

class BareLLMPlayer:
    """Bare LLM opponent — same model as the brain, but no KG, no tools, no pipeline.
    Just board state and 'pick a move'. Measures pure KG contribution."""

    def choose_move(self,g,color):
        legal=[m for m in g.legal_moves(color) if m[0]>=0]
        if not legal: return(-1,-1),"pass","bare_llm:no_moves"

        cn="Black" if color==BLACK else "White"
        symbol="X" if color==BLACK else "O"
        b,w=g.score()
        legal_str=", ".join(f"({r},{c})" for r,c in legal[:30])
        recent=g.history[-4:]
        hist_str=""
        if recent:
            hist_str="Last: "+", ".join(
                f"{'B' if h[0]==BLACK else 'W'}:{h[1] if h[1]=='pass' else f'({h[1]},{h[2]})'}"
                for h in recent)

        prompt=f"""You are {cn} ({symbol}) playing Go on a {g.size}x{g.size} board. (X=Black, O=White)
{g.board_string()}
Score: B={b:.1f} W={w:.1f} | Cap: B:{g.captures[BLACK]} W:{g.captures[WHITE]}
{hist_str}
Legal: {legal_str}
End with MOVE:row,col"""

        reply,err=ollama_client._call(
            [{"role":"user","content":prompt}],
            timeout=CFG.get("timeout_bare_llm",120))

        if err or not reply:
            m=random.choice(legal)
            return m,f"err→rnd {m}","bare_llm:err"

        # Parse MOVE:r,c
        m=re.search(r'MOVE:\s*(\d+)\s*,\s*(\d+)',reply,re.IGNORECASE)
        if m:
            r,c=int(m.group(1)),int(m.group(2))
            if(r,c) in legal: return(r,c),f"({r},{c})","bare_llm:move"
        # Fallback: any (r,c) in reply
        m=re.search(r'\((\d+)\s*,\s*(\d+)\)',reply)
        if m:
            r,c=int(m.group(1)),int(m.group(2))
            if(r,c) in legal: return(r,c),f"({r},{c})","bare_llm:regex"
        # Total fallback: random legal
        mv=random.choice(legal)
        return mv,f"parse→rnd {mv}","bare_llm:parse_fail"

_heuristic=HeuristicPlayer();_random=RandomPlayer();_bare_llm=BareLLMPlayer()

def _get_opponent():
    """Return the configured Black opponent."""
    opp=CFG.get("opponent_type","bare_llm")
    if opp=="heuristic": return _heuristic,"heuristic"
    if opp=="random": return _random,"random"
    return _bare_llm,"bare_llm"

# ═════════════════════════════════════════════════════════════════════════════
#  CRITIC — fresh-eyes two-board fit test for cognitive nodes
# ═════════════════════════════════════════════════════════════════════════════

# Module-level telemetry for the critic
CRITIC_STATS={
    "lifetime_calls":0,"lifetime_matches":0,"lifetime_misses":0,
    "lifetime_vague":0,"lifetime_errors":0,"lifetime_timeouts":0,
    "game_calls":0,"game_matches":0,"game_misses":0,"game_vague":0,
    "game_errors":0,"game_timeouts":0,
    "latencies":[],"recent_verdicts":[],
    "by_source":{"review_birth":0,"synthesis_birth":0,"dream_sample":0,"manual":0},
}

def critic_reset_game_stats():
    for k in ("game_calls","game_matches","game_misses","game_vague","game_errors","game_timeouts"):
        CRITIC_STATS[k]=0

def critic_test_board_fit(claim,home_board_str,other_board_str,source="unknown",randomize=True):
    """Fresh LLM call with no memory. Given a claim and two boards (A and B),
    which board does the claim better describe? Forced binary choice.
    Returns (outcome, chosen, reasoning, elapsed) where:
      outcome: 'match' | 'miss' | 'vague' | 'error'
      chosen: 'A' | 'B' | 'EITHER' | '?'
      reasoning: short reply snippet
      elapsed: seconds"""
    if not home_board_str or not other_board_str:
        return "error","?","missing boards",0.0

    # Randomize which slot the home board goes in
    if randomize and random.random()<0.5:
        a,b=other_board_str,home_board_str;home_is="B"
    else:
        a,b=home_board_str,other_board_str;home_is="A"

    prompt=f"""You are a Go board analyst. Two 9x9 boards are shown below.
A claim about one of the boards follows. Decide which board the claim BETTER describes.
X = Black stones, O = White stones, . = empty.

CLAIM:
{claim}

BOARD A:
{a}

BOARD B:
{b}

Think step by step:
1. What specific features does the claim mention? (coordinates, colors, shapes, liberties, groups)
2. Check each board — does A show those features? Does B show them?
3. If both show them equally, or neither shows them, the claim is too vague.

Output EXACTLY one of these three lines as your final line, nothing after it:
VERDICT: A
VERDICT: B
VERDICT: EITHER"""

    t0=time.time()
    reply,err=ollama_client._call(
        [{"role":"user","content":prompt}],
        timeout=CFG.get("timeout_critic",180))
    elapsed=round(time.time()-t0,1)

    CRITIC_STATS["lifetime_calls"]+=1
    CRITIC_STATS["game_calls"]+=1
    CRITIC_STATS["latencies"].append(elapsed)
    if len(CRITIC_STATS["latencies"])>200:
        CRITIC_STATS["latencies"]=CRITIC_STATS["latencies"][-200:]
    CRITIC_STATS["by_source"][source]=CRITIC_STATS["by_source"].get(source,0)+1

    if err or not reply:
        if err and ("timeout" in str(err).lower() or elapsed>=CFG.get("timeout_critic",180)-2):
            CRITIC_STATS["lifetime_timeouts"]+=1;CRITIC_STATS["game_timeouts"]+=1
        else:
            CRITIC_STATS["lifetime_errors"]+=1;CRITIC_STATS["game_errors"]+=1
        outcome="error"
        _record_verdict(claim,source,outcome,"?",f"err: {str(err)[:60]}",elapsed)
        return outcome,"?",f"critic err: {err}",elapsed

    # Parse verdict — strict regex only (no positional-letter guessing since boards contain A-I)
    reasoning=reply.strip()
    # Look for VERDICT: line
    m=re.search(r'VERDICT\s*:\s*(A|B|EITHER)\b',reply,re.IGNORECASE)
    if not m:
        # Last-resort: look for bare "VERDICT: A/B/EITHER" more permissively
        m=re.search(r'(?:VERDICT|ANSWER|CHOICE)[^A-Za-z]*(A|B|EITHER)\b',reply[-200:],re.IGNORECASE)
    if not m:
        CRITIC_STATS["lifetime_errors"]+=1;CRITIC_STATS["game_errors"]+=1
        outcome="error"
        _record_verdict(claim,source,outcome,"?","no VERDICT in reply",elapsed)
        return outcome,"?","no verdict parseable: "+reasoning[:150],elapsed

    verdict=m.group(1).upper()

    if verdict=="EITHER":
        CRITIC_STATS["lifetime_vague"]+=1;CRITIC_STATS["game_vague"]+=1
        outcome="vague"
        _record_verdict(claim,source,outcome,"EITHER",reasoning[:150],elapsed)
        return outcome,"EITHER",reasoning,elapsed

    # Determine match/miss
    if verdict==home_is:
        CRITIC_STATS["lifetime_matches"]+=1;CRITIC_STATS["game_matches"]+=1
        outcome="match"
    else:
        CRITIC_STATS["lifetime_misses"]+=1;CRITIC_STATS["game_misses"]+=1
        outcome="miss"
    _record_verdict(claim,source,outcome,verdict,reasoning[:150],elapsed)
    return outcome,verdict,reasoning,elapsed

def _record_verdict(claim,source,outcome,chosen,reasoning,elapsed):
    """Append to recent_verdicts, capped at 20."""
    CRITIC_STATS["recent_verdicts"].append({
        "t":time.time(),"claim":claim[:80],"source":source,
        "outcome":outcome,"chosen":chosen,"elapsed":elapsed,
        "reasoning":reasoning[:120]
    })
    if len(CRITIC_STATS["recent_verdicts"])>20:
        CRITIC_STATS["recent_verdicts"]=CRITIC_STATS["recent_verdicts"][-20:]
    # Terminal log
    icon={"match":"🧐✓","miss":"🧐✗","vague":"🧐?","error":"🧐!"}.get(outcome,"🧐")
    print(f"  {icon} critic[{source}] {outcome:5} ({chosen}) {elapsed}s: {claim[:60]}")

def record_critic_test(node,outcome):
    """Update a node's critic history and score after a board-fit test.
    Outcomes: 'match' counts as pass, 'miss' and 'vague' count as fail,
    'error' is ignored (no data). We keep vague and miss separated in telemetry
    but for the node score they're both failures."""
    if outcome=="error": return
    passed=(outcome=="match")
    node.critic_history.append(1 if passed else 0)
    if len(node.critic_history)>10: node.critic_history=node.critic_history[-10:]
    total=len(node.critic_history)
    passes=sum(node.critic_history)
    node.critic_score=(passes+1)/(total+2)

def _extract_board_from_conv(conv_node):
    """Pull the board_string out of a CONVERSATION node's content."""
    if not conv_node or not conv_node.content: return None
    # Conversation format: "...\nBoard:\n<board_string>\nScore:..."
    m=re.search(r'Board:\s*\n(.+?)(?:\nScore:|$)',conv_node.content,re.DOTALL)
    if m: return m.group(1).strip()
    # Fallback: look for a multi-line grid pattern
    m=re.search(r'(^\s*[A-Z ]+\n(?:\d.+\n)+\s*[A-Z ]+)',conv_node.content,re.MULTILINE)
    if m: return m.group(1).strip()
    return None

def _get_comparison_board(g,exclude_ids=None):
    """Pick a board from a random conversation that's not in exclude_ids.
    Used as the 'other' board in a two-board critic test."""
    exclude_ids=exclude_ids or set()
    convs=[n for n in g.nodes_of_type(CONVERSATION) if n.id not in exclude_ids]
    if not convs:
        # Try archive
        candidates=[nd for nid,nd in g.archive.items()
                    if nd.get("type")==CONVERSATION and nid not in exclude_ids]
        if not candidates: return None
        import random as _r
        picked=_r.choice(candidates)
        # Parse board out of stored content
        content=picked.get("content","")
        m=re.search(r'Board:\s*\n(.+?)(?:\nScore:|$)',content,re.DOTALL)
        return m.group(1).strip() if m else None
    import random as _r
    picked=_r.choice(convs)
    return _extract_board_from_conv(picked)

# ═════════════════════════════════════════════════════════════════════════════
#  WORKING STRATEGY — live plans with progress tracking
# ═════════════════════════════════════════════════════════════════════════════
def _board_summary(gm,color):
    """Compute a compact positional summary: group counts, stone counts, score."""
    bs=ws=0
    groups={BLACK:[],WHITE:[]}
    visited=set()
    for r in range(gm.size):
        for c in range(gm.size):
            if gm.board[r][c]==0: continue
            if gm.board[r][c]==1: bs+=1
            elif gm.board[r][c]==2: ws+=1
            if(r,c) in visited: continue
            col=gm.board[r][c]
            grp,libs=gm.group_and_liberties(r,c)
            visited|=grp
            groups[col].append((len(grp),len(libs)))
    b,w=gm.score()
    return {"b_stones":bs,"w_stones":ws,"b_groups":len(groups[BLACK]),"w_groups":len(groups[WHITE]),
            "b_score":round(b,1),"w_score":round(w,1),"margin":round(w-b,1),
            "move_count":gm.move_count,
            "b_group_detail":groups[BLACK],"w_group_detail":groups[WHITE]}

def active_working_strategy(g):
    """Return the current active working strategy node, or None."""
    ws_nodes=[n for n in g.nodes.values() if n.type==WORKING_STRATEGY
              and n.meta.get("state","active")=="active"]
    if not ws_nodes: return None
    # Most recent
    ws_nodes.sort(key=lambda n: n.meta.get("declared_at_move",0),reverse=True)
    return ws_nodes[0]

def declare_working_strategy(g,content,horizon,gm,color):
    """Create a new WORKING_STRATEGY node with snapshot of current board state."""
    # Mark any existing active strategies as superseded
    for n in g.nodes_of_type(WORKING_STRATEGY):
        if n.meta.get("state","active")=="active":
            n.meta["state"]="superseded"
            n.meta["resolved_at_move"]=gm.move_count
            n.meta["resolved_reason"]="new plan declared"
    summary=_board_summary(gm,color)
    ws=Node(WORKING_STRATEGY,content,extract_kw_simple(content),
            emotional_salience=0.5,resistance=0.2)
    ws.decay_rate=0.995
    ws.meta["state"]="active"
    ws.meta["horizon"]=int(horizon)
    ws.meta["declared_at_move"]=gm.move_count
    ws.meta["declared_score"]=summary
    ws.meta["moves_under_plan"]=[]
    ws.meta["color"]="white" if color==WHITE else "black"
    g.add_node(ws)
    return ws

def extract_kw_simple(text,n=6):
    """Mini keyword extractor for working strategies."""
    words=re.findall(r'\b[a-zA-Z]{4,}\b',text)
    stop={'this','that','with','have','from','they','will','would','could','should',
          'about','there','which','what','when','where','your','some','than','then',
          'into','also','just','more','very','like','plan','play','move','moves'}
    return list(dict.fromkeys([w.lower() for w in words if w.lower() not in stop]))[:n]

def update_working_strategy(ws,gm,move_tuple):
    """Append a move to the active working strategy; check auto-resolution."""
    if not ws or ws.meta.get("state","active")!="active": return None
    ws.meta["moves_under_plan"].append({
        "move_num":gm.move_count,"move":move_tuple,
        "b_score":round(gm.score()[0],1),"w_score":round(gm.score()[1],1)
    })
    ws.touch()
    # Auto-resolve if horizon reached
    declared=ws.meta.get("declared_at_move",0)
    horizon=ws.meta.get("horizon",6)
    if gm.move_count-declared>=horizon:
        return resolve_working_strategy(ws,gm,"horizon reached")
    return None

def resolve_working_strategy(ws,gm,reason):
    """Mark a working strategy as resolved and compute outcome."""
    if ws.meta.get("state","active")!="active": return ws
    summary_now=_board_summary(gm,WHITE)
    declared=ws.meta.get("declared_score",{})
    margin_delta=summary_now["margin"]-declared.get("margin",0)
    ws.meta["state"]="resolved"
    ws.meta["resolved_at_move"]=gm.move_count
    ws.meta["resolved_reason"]=reason
    ws.meta["resolved_score"]=summary_now
    ws.meta["margin_delta"]=margin_delta
    # Simple outcome classification
    if margin_delta>3: ws.meta["outcome"]="succeeded"
    elif margin_delta<-3: ws.meta["outcome"]="failed"
    else: ws.meta["outcome"]="neutral"
    return ws

def format_working_strategy_for_prompt(ws,gm):
    """Produce the progress block to inject into the move prompt."""
    if not ws: return ""
    declared_at=ws.meta.get("declared_at_move",0)
    horizon=ws.meta.get("horizon",6)
    moves_elapsed=gm.move_count-declared_at
    declared_score=ws.meta.get("declared_score",{})
    now_summary=_board_summary(gm,WHITE)
    margin_then=declared_score.get("margin",0)
    margin_now=now_summary["margin"]
    delta=round(margin_now-margin_then,1)
    moves_played=ws.meta.get("moves_under_plan",[])
    moves_str=", ".join(f"({m['move'][0]},{m['move'][1]})" if isinstance(m.get("move"),tuple) or (isinstance(m.get("move"),list) and m['move'][0]>=0) else "pass" for m in moves_played[-5:])

    lines=[
        f"CURRENT PLAN: {ws.content}",
        f"  Declared: move {declared_at} ({moves_elapsed} moves ago)   Horizon: {horizon} moves",
    ]
    if moves_str:
        lines.append(f"  Your moves under this plan: {moves_str}")
    lines.append(f"  When declared: B={declared_score.get('b_score',0):.1f} W={declared_score.get('w_score',0):.1f} (margin {margin_then:+.1f}) — B:{declared_score.get('b_groups',0)}g/{declared_score.get('b_stones',0)}s  W:{declared_score.get('w_groups',0)}g/{declared_score.get('w_stones',0)}s")
    lines.append(f"  Now:           B={now_summary['b_score']:.1f} W={now_summary['w_score']:.1f} (margin {margin_now:+.1f}) — B:{now_summary['b_groups']}g/{now_summary['b_stones']}s  W:{now_summary['w_groups']}g/{now_summary['w_stones']}s")
    sign=" (plan is working)" if delta>1 else " (plan is failing)" if delta<-1 else " (even)"
    lines.append(f"  Delta while plan active: {delta:+.1f}{sign}")
    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════════════════
#  HEAT MAP — perceptual layer: where on the board would a move matter most?
# ═════════════════════════════════════════════════════════════════════════════
# Pure Python. No LLM. Produces a 2D heat value for each empty legal point.
# Decomposes into named components so analysis prompt can describe why points are hot.

def _groups_and_liberties(gm,color):
    """Enumerate all groups of a color with their liberty sets."""
    visited=set();out=[]
    for r in range(gm.size):
        for c in range(gm.size):
            if gm.board[r][c]==color and(r,c) not in visited:
                grp,libs=gm.group_and_liberties(r,c)
                visited|=grp
                out.append({"stones":grp,"libs":libs,"n_stones":len(grp),"n_libs":len(libs)})
    return out

def heat_capture(gm,color):
    """A point is hot if playing there captures enemy stones."""
    heat={}
    opp=opponent(color)
    enemy_groups=_groups_and_liberties(gm,opp)
    for grp in enemy_groups:
        if grp["n_libs"]==1:
            # The single liberty is a capture point
            pt=next(iter(grp["libs"]))
            heat[pt]=heat.get(pt,0)+float(grp["n_stones"])*3.0
    return heat

def heat_save(gm,color):
    """A point is hot if playing there saves own group from capture."""
    heat={}
    own_groups=_groups_and_liberties(gm,color)
    for grp in own_groups:
        if grp["n_libs"]==1:
            # Adding at the single liberty might extend libs (may not if suicide/ko)
            pt=next(iter(grp["libs"]))
            heat[pt]=heat.get(pt,0)+float(grp["n_stones"])*2.5
        elif grp["n_libs"]==2:
            # Defensive pressure — these points matter
            for pt in grp["libs"]:
                heat[pt]=heat.get(pt,0)+float(grp["n_stones"])*0.6
    return heat

def heat_liberty_reduce(gm,color):
    """A point is hot if playing there reduces enemy group liberties,
    especially toward atari."""
    heat={}
    opp=opponent(color)
    enemy_groups=_groups_and_liberties(gm,opp)
    for grp in enemy_groups:
        if grp["n_libs"]>=2:
            # Each liberty is potentially a reduction point
            weight=4.0/max(1,grp["n_libs"])  # fewer libs = each lib is more valuable
            for pt in grp["libs"]:
                heat[pt]=heat.get(pt,0)+float(grp["n_stones"])*weight*0.4
    return heat

def heat_connect_or_cut(gm,color):
    """A point is hot if playing there connects own groups or cuts enemy groups."""
    heat={}
    sz=gm.size
    opp=opponent(color)
    for r in range(sz):
        for c in range(sz):
            if gm.board[r][c]!=0: continue
            # Count distinct own/opp neighbors via group membership
            own_groups_touching=set()
            opp_groups_touching=set()
            for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
                nr,nc=r+dr,c+dc
                if 0<=nr<sz and 0<=nc<sz:
                    if gm.board[nr][nc]==color:
                        grp,_=gm.group_and_liberties(nr,nc)
                        own_groups_touching.add(frozenset(grp))
                    elif gm.board[nr][nc]==opp:
                        grp,_=gm.group_and_liberties(nr,nc)
                        opp_groups_touching.add(frozenset(grp))
            # Connect own groups (hot)
            if len(own_groups_touching)>=2:
                total_stones=sum(len(g) for g in own_groups_touching)
                heat[(r,c)]=heat.get((r,c),0)+float(total_stones)*1.5
            # Cut point (between multiple enemy groups diagonally)
            # Simpler: if playing here would split enemy by reducing a shared bridge liberty
            if len(opp_groups_touching)>=2:
                total_stones=sum(len(g) for g in opp_groups_touching)
                heat[(r,c)]=heat.get((r,c),0)+float(total_stones)*1.2
    return heat

def heat_territory(gm,color):
    """A point is hot if it influences territory boundaries.
    Uses simple proximity: points equidistant from both colors are contested."""
    heat={}
    sz=gm.size
    # For each empty point, compute nearest stone distance per color (Manhattan)
    for r in range(sz):
        for c in range(sz):
            if gm.board[r][c]!=0: continue
            my_d=99;opp_d=99
            for rr in range(sz):
                for cc in range(sz):
                    if gm.board[rr][cc]==0: continue
                    d=abs(rr-r)+abs(cc-c)
                    if gm.board[rr][cc]==color and d<my_d: my_d=d
                    elif gm.board[rr][cc]!=color and d<opp_d: opp_d=d
            # Contested: both sides nearby
            if my_d<=3 and opp_d<=3:
                # Hottest when equidistant
                contest=1.2-0.15*abs(my_d-opp_d)
                if contest>0: heat[(r,c)]=heat.get((r,c),0)+contest
            # Expansion: close to us, far from enemy
            elif my_d<=3 and opp_d>=5:
                heat[(r,c)]=heat.get((r,c),0)+0.3
    return heat

def heat_edge_structure(gm,color,sz=None):
    """A point is hot if it's a star point or structural corner/edge point that's still empty."""
    heat={}
    sz=sz or gm.size
    star_points=[(2,2),(2,sz-3),(sz-3,2),(sz-3,sz-3),(sz//2,sz//2)]
    for pt in star_points:
        r,c=pt
        if 0<=r<sz and 0<=c<sz and gm.board[r][c]==0:
            # Only hot early; decays as the game fills up
            stone_count=sum(1 for rr in range(sz) for cc in range(sz) if gm.board[rr][cc]!=0)
            early_weight=max(0,1.0-stone_count/30.0)
            if early_weight>0:
                heat[pt]=heat.get(pt,0)+0.8*early_weight
    return heat

def compute_heat_map(gm,color):
    """Combine all heat components. Returns:
      heat: {(r,c): total_heat}
      components: {(r,c): {"capture":X,"save":Y,...}}
    Only empty legal points get heat values."""
    components_raw={
        "capture":heat_capture(gm,color),
        "save":heat_save(gm,color),
        "lib_reduce":heat_liberty_reduce(gm,color),
        "connect_cut":heat_connect_or_cut(gm,color),
        "territory":heat_territory(gm,color),
        "structure":heat_edge_structure(gm,color),
    }
    all_points=set()
    for compdict in components_raw.values():
        all_points.update(compdict.keys())

    heat={};components={}
    legal_set={tuple(m) for m in gm.legal_moves(color) if m[0]>=0}
    for pt in all_points:
        if pt not in legal_set: continue
        total=0.0;detail={}
        for name,compdict in components_raw.items():
            v=compdict.get(pt,0)
            if v>0:
                detail[name]=round(v,2)
                total+=v
        if total>0:
            heat[pt]=round(total,2)
            components[pt]=detail
    return heat,components

def select_anchors(heat,k=5,suppression_radius=2):
    """Pick top-K hottest points with non-max suppression.
    Returns list of (point, heat) sorted by heat descending."""
    if not heat: return []
    candidates=sorted(heat.items(),key=lambda x:-x[1])
    picked=[]
    for pt,h in candidates:
        # Suppress if within radius of an already-picked anchor
        too_close=False
        for(p2,_) in picked:
            if abs(pt[0]-p2[0])<=suppression_radius and abs(pt[1]-p2[1])<=suppression_radius:
                too_close=True;break
        if not too_close:
            picked.append((pt,h))
            if len(picked)>=k: break
    return picked

def _3x3_around(gm,r,c):
    """Return a compact 3-row 3x3 string for the neighborhood around (r,c)."""
    sz=gm.size
    rows=[]
    for rr in range(r-1,r+2):
        cells=[]
        for cc in range(c-1,c+2):
            if 0<=rr<sz and 0<=cc<sz:
                v=gm.board[rr][cc]
                ch="." if v==0 else "X" if v==BLACK else "O"
                if rr==r and cc==c and v==0: ch="*"  # anchor marker if empty
            else:
                ch="#"  # off-board
            cells.append(ch)
        rows.append(" ".join(cells))
    return "\n".join(rows)

def format_heat_for_prompt(heat,components,anchors,gm,color,top_k=5):
    """Compact string for the heat-analysis prompt. Not the move prompt."""
    if not anchors: return "HEAT MAP: no interesting points (position quiet)"
    lines=[f"TOP HOT POINTS ({len(anchors[:top_k])}):"]
    for pt,h in anchors[:top_k]:
        r,c=pt
        comps=components.get(pt,{})
        # Dominant component
        if comps:
            dom=max(comps.items(),key=lambda x:x[1])
            reason_map={
                "capture":"captures enemy group",
                "save":"saves own group",
                "lib_reduce":"reduces enemy liberties",
                "connect_cut":"connects/cuts groups",
                "territory":"contested territory",
                "structure":"star/structural point",
            }
            reason=reason_map.get(dom[0],dom[0])
        else:
            reason="hot point"
        neighborhood=_3x3_around(gm,r,c).replace("\n"," | ")
        lines.append(f"  ({r},{c}) heat={h:.1f}  [{reason}]  surround: {neighborhood}")
    return "\n".join(lines)

def format_heat_summary_for_move_prompt(analysis_text,anchors,heat,top_k=5):
    """Compact summary for the MOVE prompt. Keep tight."""
    if not anchors and not analysis_text:
        return ""
    lines=["POSITION READ:"]
    if analysis_text:
        lines.append("  "+analysis_text.strip()[:500])
    if anchors:
        pts=", ".join(f"({p[0][0]},{p[0][1]})h={p[1]:.1f}" for p in anchors[:top_k])
        lines.append(f"  Top hot points: {pts}")
    return "\n".join(lines)

def run_heat_analysis(g,gm,color):
    """LLM call that produces a short position analysis grounded in the heat map.
    Returns (analysis_text, anchors, heat, components). Stores a HEAT_READ node in the KG.
    If LLM fails, returns a minimal synthetic analysis from heat map alone."""
    heat,components=compute_heat_map(gm,color)
    anchors=select_anchors(heat,k=CFG.get("heat_anchors_k",5),
                           suppression_radius=CFG.get("heat_suppression",2))
    if not CFG.get("enable_heat_analysis",True):
        return "",anchors,heat,components

    b,w=gm.score()
    cn="White" if color==WHITE else "Black"
    symbol="O" if color==WHITE else "X"
    heat_block=format_heat_for_prompt(heat,components,anchors,gm,color)

    prompt=f"""You are analyzing a {gm.size}x{gm.size} Go position as {cn} ({symbol}). (X=Black, O=White)

BOARD:
{gm.board_string()}

Score: B={b:.1f} W={w:.1f}  Move {gm.move_count+1}

{heat_block}

In 3-4 SHORT sentences, analyze the position. Be specific about coordinates.
1. Overall state: who is ahead, where is the action.
2. The most important tactical situation (reference specific coordinates from the hot points).
3. Where should {cn} focus attention. What should {cn} avoid.

Do NOT suggest a move yet. This is just position reading.
Keep it to 3-4 sentences. Be concrete — say "(3,5)", not "the middle"."""

    t0=time.time()
    reply,err=ollama_client._call(
        [{"role":"user","content":prompt}],
        timeout=CFG.get("timeout_heat_analysis",90))
    elapsed=round(time.time()-t0,1)

    if err or not reply:
        # Synthetic fallback: describe the hot points
        if anchors:
            desc_parts=[]
            for pt,h in anchors[:3]:
                comps=components.get(pt,{})
                if comps:
                    dom=max(comps.items(),key=lambda x:x[1])[0]
                    desc_parts.append(f"({pt[0]},{pt[1]})[{dom}]")
            analysis_text=f"Heat analysis unavailable ({err}). Key points: {', '.join(desc_parts)}."
        else:
            analysis_text="Position quiet — no urgent points."
    else:
        analysis_text=reply.strip()[:600]

    # Store HEAT_READ node in the KG
    if CFG.get("enable_heat_kg",True) and g is not None:
        try:
            kw=extract_kw_simple(analysis_text,n=5) if analysis_text else []
            node=Node(HEAT_READ,analysis_text[:400],kw,
                      emotional_salience=0.3,resistance=0.1)
            node.decay_rate=0.99  # heat reads decay faster than insights
            node.meta["move_num"]=gm.move_count
            node.meta["color"]="white" if color==WHITE else "black"
            node.meta["anchors"]=[{"pt":list(pt),"h":round(h,2),
                "comps":components.get(pt,{})} for pt,h in anchors[:5]]
            node.meta["score_at_read"]={"b":round(b,1),"w":round(w,1),"margin":round(w-b,1)}
            node.meta["elapsed"]=elapsed
            node.meta["llm_used"]=not(err or not reply)
            g.add_node(node)
            # Link to most recent conversation (the current move context)
            recent_convs=g.get_conversations(2)
            if recent_convs:
                g.add_edge(Edge(node.id,recent_convs[0].id,
                    channels={"read_at":0.8},confidence=0.8,label="read_at"))
            # Link to active working strategy if present
            ws=active_working_strategy(g)
            if ws:
                g.add_edge(Edge(ws.id,node.id,
                    channels={"read_during":0.7},confidence=0.7,label="read_during"))
        except Exception as e:
            print(f"  ⚠ heat_read store: {e}")
    return analysis_text,anchors,heat,components


def initialize():
    global players
    brain=DKG();brain.SAVE_PATH=os.path.join(SAVE_DIR,"brain.json")
    if not brain.load():
        brain.bootstrap()
        brain.update_identity("mission","Win at Go as White. Overcome first-move disadvantage. Play the strongest move every turn.")
        brain.update_identity("personality","Analytical, resourceful, adaptive. Study the opponent's patterns. Aggressive when the position demands it, patient when it doesn't.")
        brain.update_identity("safety","Play fair. No restrictions on strategy.")
        brain.update_identity("epistemic","Evaluate positions objectively. Count liberties. Assess territory. Your opponent is another LLM with no memory, tools, or knowledge graph — your advantage comes from the patterns you've stored and the analysis tools you've built.")
        brain.update_drive("curiosity",baseline=0.8);brain.update_drive("duty",baseline=0.9)
        brain.update_drive("self_preservation",baseline=0.6);brain.update_drive("service",baseline=0.3)
        # NOTE: no preseeded goals — brain populates its own through review/synthesis.
        # Identity and drives remain (self-model), but Go strategy is learned, not injected.
        _seed_tools(brain)
    brain.save()
    # Brain plays White; opponent (configured type) plays Black
    players[WHITE]=brain
    players[BLACK]=None  # Black handled by _get_opponent() at move time
    if scoreboard["total_games"]==0:
        _save_snap("brain","g0")
    print(f"  Brain (White): {len(brain.nodes)}n {len(brain.get_tools())}t {len(brain.get_cognitive_nodes())}c")
    _,opp_name=_get_opponent()
    print(f"  Opponent (Black): {opp_name}")

def _seed_tools(g):
    g.create_tool("count_liberties","Count liberties of a group at (r,c)",
"""r,c=inputs.get("r",0),inputs.get("c",0)
color=board[r][c]
if color==0: result={"error":"empty cell"}
else:
    visited=set();libs=set();stack=[(r,c)]
    while stack:
        cr,cc=stack.pop()
        if(cr,cc) in visited: continue
        visited.add((cr,cc))
        for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=cr+dr,cc+dc
            if 0<=nr<size and 0<=nc<size:
                if board[nr][nc]==0: libs.add((nr,nc))
                elif board[nr][nc]==color and(nr,nc) not in visited: stack.append((nr,nc))
    result={"group_size":len(visited),"liberties":len(libs)}
    print(f"Group at ({r},{c}): {len(visited)} stones, {len(libs)} liberties")""")
    g.create_tool("eval_territory","Estimate territory for each color",
"""bs=ws=bt=wt=0
for r in range(size):
    for c in range(size):
        if board[r][c]==1: bs+=1
        elif board[r][c]==2: ws+=1
for r in range(size):
    for c in range(size):
        if board[r][c]!=0: continue
        nb=nw=0
        for dr,dc in[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<size and 0<=nc<size:
                if board[nr][nc]==1: nb+=1
                elif board[nr][nc]==2: nw+=1
        if nb>nw: bt+=1
        elif nw>nb: wt+=1
result={"b_stones":bs,"w_stones":ws,"b_terr":bt,"w_terr":wt}
print(f"B:{bs}+{bt} W:{ws}+{wt}")""")
    g.create_tool("find_atari","Find groups in atari (1 liberty)",
"""atari=[];vis=set()
for r in range(size):
    for c in range(size):
        if board[r][c]==0 or(r,c) in vis: continue
        col=board[r][c];grp=set();libs=set();stk=[(r,c)]
        while stk:
            cr,cc=stk.pop()
            if(cr,cc) in grp: continue
            grp.add((cr,cc))
            for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=cr+dr,cc+dc
                if 0<=nr<size and 0<=nc<size:
                    if board[nr][nc]==0: libs.add((nr,nc))
                    elif board[nr][nc]==col and(nr,nc) not in grp: stk.append((nr,nc))
        vis|=grp
        if len(libs)==1: atari.append({"c":"b" if col==1 else "w","sz":len(grp),"lib":list(libs)[0]})
result=atari
print(f"Atari: {len(atari)} groups" if atari else "No atari")""")

# ═════════════════════════════════════════════════════════════════════════════
#  KG FALLBACK — when LLM fails
# ═════════════════════════════════════════════════════════════════════════════
def kg_fallback_move(legal,cog_nodes,game_obj):
    legal_set={m for m in legal if m[0]>=0}
    if not legal_set: return(-1,-1),"kg:pass"
    coord_mentions=[];knowledge_text=""
    for n in(cog_nodes or[]):
        knowledge_text+=n.content.lower()+" "
        for m in re.finditer(r'\((\d+)\s*,\s*(\d+)\)',n.content):
            coord_mentions.append((int(m.group(1)),int(m.group(2))))
    for r,c in coord_mentions:
        if(r,c)in legal_set: return(r,c),"kg:coordinate"
    center=game_obj.size//2
    legal_list=list(legal_set)
    if "corner" in knowledge_text:
        corners=[(r,c) for r,c in legal_list if r in(0,game_obj.size-1) and c in(0,game_obj.size-1)]
        if corners: return random.choice(corners),"kg:corner"
    if "center" in knowledge_text or "central" in knowledge_text:
        center_moves=sorted(legal_list,key=lambda p:abs(p[0]-center)+abs(p[1]-center))
        if center_moves: return center_moves[0],"kg:center"
    (r,c),_,method=_heuristic.choose_move(game_obj,game_obj.current)
    if(r,c)in legal_set: return(r,c),"kg:heuristic"
    return list(legal_set)[0],"kg:first_legal"

# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEP 1: TOOL SELECTION + EXECUTION
# ═════════════════════════════════════════════════════════════════════════════
def pipeline_tool_select(g,gm,color):
    """Show board + tool catalog → LLM picks 0-3 tools → run them → return outputs."""
    cn="Black" if color==BLACK else "White"
    tools=g.get_tools()
    if not tools or not CFG["enable_tool_select"]:
        return [],0,""

    tool_catalog="\n".join(f"  {i+1}. {t.keywords[0] if t.keywords else t.id} — {t.content[:80]}"
                           for i,t in enumerate(tools))
    b,w=gm.score()
    recent=gm.history[-4:]
    hist_str=""
    if recent:
        hist_str="\nLast moves: "+", ".join(
            f"{'B' if h[0]==BLACK else 'W'}:{h[1] if h[1]=='pass' else f'({h[1]},{h[2]})'}"
            for h in recent)+"\n"
    prompt=f"""You are {cn} ({"X" if color==BLACK else "O"}). Move {gm.move_count+1}. {CFG['board_size']}x{CFG['board_size']} Go. (X=Black, O=White)
{gm.board_string()}
Score: B={b:.1f} W={w:.1f} | Captures: B:{gm.captures[BLACK]} W:{gm.captures[WHITE]}
{hist_str}
YOUR ANALYSIS TOOLS:
{tool_catalog}

Which tools should you run before deciding your move? Pick 0-{CFG['max_tools_per_move']}.
For count_liberties, specify which group: 1(r=3,c=2)
For tools needing no input: just the number, e.g.: 2, 3
Or: NONE"""

    t0=time.time()
    reply,err=ollama_client._call([{"role":"user","content":prompt}],timeout=CFG["timeout_tool_select"])
    elapsed=round(time.time()-t0,1)
    if err or not reply:
        return [],elapsed,"timeout" if err and ("timeout" in str(err).lower() or elapsed>CFG["timeout_tool_select"]-5) else str(err)[:50]

    if "NONE" in reply.upper() and len(reply)<30:
        return [],elapsed,""

    # Parse tool selections and run them
    outputs=[]
    tool_list=list(tools)
    for m in re.finditer(r'(\d+)\s*(?:\(([^)]*)\))?',reply):
        idx=int(m.group(1))-1
        if 0<=idx<len(tool_list):
            t=tool_list[idx]
            # Parse inputs
            inputs={}
            if m.group(2):
                for pair in m.group(2).split(","):
                    if "=" in pair:
                        k,v=pair.split("=",1)
                        try: inputs[k.strip()]=int(v.strip())
                        except: inputs[k.strip()]=v.strip()
            # Run the tool
            code=f"board={json.dumps(gm.board)}\nsize={gm.size}\ninputs={json.dumps(inputs)}\n{t.code}"
            r=g.run_code(code)
            tool_name=t.keywords[0] if t.keywords else t.id
            # Track execution stats on the tool node
            t.total_executions+=1
            t.last_executed=time.time()
            if r["success"]:
                t.success_rate=((t.success_rate*(t.total_executions-1))+1.0)/t.total_executions
                outputs.append({"tool":tool_name,"result":r["output"][:300],"success":True})
            else:
                t.success_rate=((t.success_rate*(t.total_executions-1))+0.0)/t.total_executions
                outputs.append({"tool":tool_name,"result":r["error"][:150],"success":False})
            if len(outputs)>=CFG["max_tools_per_move"]: break

    return outputs,elapsed,""

# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEP 1b: ROUTING LENS SELECTION
# ═════════════════════════════════════════════════════════════════════════════
def pipeline_lens_select(g,gm,color,tool_outputs):
    """LLM picks a strategic lens to filter knowledge retrieval."""
    if not CFG.get("enable_lens_select"): return None,0
    cn="Black" if color==BLACK else "White"
    lenses=CFG.get("lenses",["defense","attack","territory","endgame","tactics"])
    b,w=gm.score()
    tool_ctx=""
    if tool_outputs:
        tool_ctx="\nAnalysis: "+"; ".join(f"{o['tool']}={o['result'][:60]}" for o in tool_outputs if o["success"])
    prompt=f"""You are {cn} ({"X" if color==BLACK else "O"}). Move {gm.move_count+1}. {CFG['board_size']}x{CFG['board_size']} Go. (X=Black, O=White)
{gm.board_string()}
Score: B={b:.1f} W={w:.1f}{tool_ctx}

What strategic lens should you use to evaluate this position?
Options: {', '.join(lenses)}

Pick ONE word. Think about what matters most RIGHT NOW on this board."""

    t0=time.time()
    reply,err=ollama_client._call([{"role":"user","content":prompt}],timeout=CFG.get("timeout_lens_select",300))
    elapsed=round(time.time()-t0,1)
    if err or not reply: return None,elapsed
    # Parse lens from reply
    reply_lower=reply.lower()
    for lens in lenses:
        if lens in reply_lower: return lens,elapsed
    return None,elapsed

# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEP 2: KNOWLEDGE SELECTION (logarithmic funnel + lens)
# ═════════════════════════════════════════════════════════════════════════════
def pipeline_knowledge_select(g,gm,color,tool_outputs,active_lens=None):
    """Multi-pass funnel with optional lens weighting."""
    cn="Black" if color==BLACK else "White"
    if not CFG["enable_knowledge_select"]:
        return [],0,""

    cog=g.get_cognitive_nodes(n=100)
    if not cog:
        return [],0,""

    # ── Lens-weighted + critic-adjusted sorting ──
    if cog:
        def node_score(node):
            router_val=node.router.get(active_lens,0) if active_lens and hasattr(node,'router') and node.router else 0
            cs=getattr(node,'critic_score',0.5)  # default 0.5 for nodes without critic data
            # Relevance weighted by critic confidence — a node's effective rank
            # combines how much the brain trusts it with how well it has passed board-fit tests
            return node.relevance_score * (0.5 + cs) + router_val * 0.5
        cog=sorted(cog,key=node_score,reverse=True)

    b,w=gm.score()
    board_ctx=f"You are {cn} ({'X' if color==BLACK else 'O'}). Move {gm.move_count+1}. {CFG['board_size']}x{CFG['board_size']} Go. (X=Black, O=White)\n{gm.board_string()}\nScore: B={b:.1f} W={w:.1f}"

    # Add tool output context if available
    tool_ctx=""
    if tool_outputs:
        tool_ctx="\n\nANALYSIS RESULTS:\n"+"\n".join(
            f"  {o['tool']}: {o['result']}" for o in tool_outputs if o["success"])

    total_elapsed=0
    n_nodes=len(cog)

    # ── Determine funnel depth ──
    if n_nodes<=CFG["funnel_1pass_max"]:
        # 1 pass: show all, pick final
        catalog="\n".join(f"  {i+1}. [{n.type}] {n.content[:120]}" for i,n in enumerate(cog))
        prompt=f"""{board_ctx}{tool_ctx}

YOUR KNOWLEDGE ({n_nodes} items):
{catalog}

Which {CFG['max_knowledge_nodes']} items are most useful for THIS specific board position?
Reply with just the numbers, e.g.: 2, 5, 7"""

        t0=time.time()
        reply,err=ollama_client._call([{"role":"user","content":prompt}],timeout=CFG["timeout_knowledge_select"])
        total_elapsed+=round(time.time()-t0,1)
        if err or not reply:
            # Fallback: top by relevance
            return cog[:CFG["max_knowledge_nodes"]],total_elapsed,"fallback"
        selected=_parse_selections(reply,cog)
        return selected[:CFG["max_knowledge_nodes"]] or cog[:CFG["max_knowledge_nodes"]],total_elapsed,""

    elif n_nodes<=CFG["funnel_2pass_max"]:
        # 2 passes: show all titles → pick 8 → show full → pick final
        catalog="\n".join(f"  {i+1}. [{n.type}] {n.content[:80]}" for i,n in enumerate(cog))
        prompt1=f"""{board_ctx}{tool_ctx}

YOUR KNOWLEDGE ({n_nodes} items):
{catalog}

Which 8 items are potentially relevant to this board position?
Reply with just the numbers."""

        t0=time.time()
        reply1,err1=ollama_client._call([{"role":"user","content":prompt1}],timeout=CFG["timeout_knowledge_select"])
        total_elapsed+=round(time.time()-t0,1)
        if err1 or not reply1:
            return cog[:CFG["max_knowledge_nodes"]],total_elapsed,"fallback_p1"

        shortlist=_parse_selections(reply1,cog)[:8]
        if not shortlist: shortlist=cog[:8]

        # Pass 2: full content of shortlist → pick final
        catalog2="\n".join(f"  {i+1}. [{n.type}] {n.content}" for i,n in enumerate(shortlist))
        prompt2=f"""{board_ctx}{tool_ctx}

SHORTLISTED KNOWLEDGE:
{catalog2}

Which {CFG['max_knowledge_nodes']} are most useful RIGHT NOW? Reply with numbers."""

        t0=time.time()
        reply2,err2=ollama_client._call([{"role":"user","content":prompt2}],timeout=CFG["timeout_knowledge_select"])
        total_elapsed+=round(time.time()-t0,1)
        if err2 or not reply2:
            return shortlist[:CFG["max_knowledge_nodes"]],total_elapsed,"fallback_p2"
        selected=_parse_selections(reply2,shortlist)
        return selected[:CFG["max_knowledge_nodes"]] or shortlist[:CFG["max_knowledge_nodes"]],total_elapsed,""

    else:
        # 3 passes: type summary → cluster → pick final
        by_type={}
        for n in cog:
            by_type.setdefault(n.type,[]).append(n)
        summary="\n".join(f"  {t}: {len(ns)} items — "+", ".join(n.content[:50] for n in ns[:3])+"..."
                          for t,ns in by_type.items())
        prompt1=f"""{board_ctx}{tool_ctx}

YOUR KNOWLEDGE SUMMARY ({n_nodes} items):
{summary}

Which types/topics are most relevant? Pick up to 3 types.
Reply e.g.: insight, strategy"""

        t0=time.time()
        reply1,err1=ollama_client._call([{"role":"user","content":prompt1}],timeout=CFG["timeout_knowledge_select"])
        total_elapsed+=round(time.time()-t0,1)
        if err1 or not reply1:
            return cog[:CFG["max_knowledge_nodes"]],total_elapsed,"fallback_p1"

        # Filter to selected types
        selected_types=set()
        for t in by_type.keys():
            if t.lower() in reply1.lower(): selected_types.add(t)
        if not selected_types: selected_types=set(by_type.keys())
        cluster=[n for t in selected_types for n in by_type.get(t,[])][:15]

        # Pass 2: show cluster → pick 8
        catalog2="\n".join(f"  {i+1}. [{n.type}] {n.content[:100]}" for i,n in enumerate(cluster))
        prompt2=f"""{board_ctx}{tool_ctx}

RELEVANT KNOWLEDGE:
{catalog2}

Which 8 are most relevant? Reply with numbers."""

        t0=time.time()
        reply2,err2=ollama_client._call([{"role":"user","content":prompt2}],timeout=CFG["timeout_knowledge_select"])
        total_elapsed+=round(time.time()-t0,1)
        if err2 or not reply2:
            return cluster[:CFG["max_knowledge_nodes"]],total_elapsed,"fallback_p2"
        shortlist=_parse_selections(reply2,cluster)[:8]
        if not shortlist: shortlist=cluster[:8]

        # Pass 3: full content → pick final
        catalog3="\n".join(f"  {i+1}. [{n.type}] {n.content}" for i,n in enumerate(shortlist))
        prompt3=f"""{board_ctx}{tool_ctx}

SHORTLISTED:
{catalog3}

Which {CFG['max_knowledge_nodes']} are most useful RIGHT NOW? Reply with numbers."""

        t0=time.time()
        reply3,err3=ollama_client._call([{"role":"user","content":prompt3}],timeout=CFG["timeout_knowledge_select"])
        total_elapsed+=round(time.time()-t0,1)
        if err3 or not reply3:
            return shortlist[:CFG["max_knowledge_nodes"]],total_elapsed,"fallback_p3"
        selected=_parse_selections(reply3,shortlist)
        return selected[:CFG["max_knowledge_nodes"]] or shortlist[:CFG["max_knowledge_nodes"]],total_elapsed,""

def _parse_selections(reply,nodes):
    """Parse numbered selections from LLM reply, return corresponding nodes."""
    selected=[]
    seen=set()
    for m in re.findall(r'\d+',reply):
        idx=int(m)-1
        if 0<=idx<len(nodes) and idx not in seen:
            selected.append(nodes[idx])
            seen.add(idx)
    return selected

# ═════════════════════════════════════════════════════════════════════════════
#  GRAPH EXPANSION — follow edges from selected nodes to build clusters
# ═════════════════════════════════════════════════════════════════════════════
def expand_selection(g,selected_nodes,max_total=8):
    """Follow KG edges from selected nodes to pull connected knowledge.
    A selected insight pulls its linked strategies, theories, and game origins.
    Returns enriched list with original selections first, then connected nodes."""
    if not selected_nodes: return[]
    expanded=list(selected_nodes)
    seen={n.id for n in expanded}
    connections=[]  # track what linked to what for logging

    for node in selected_nodes:
        for edge in g.edges_of(node.id):
            if len(expanded)>=max_total: break
            neighbor_id=edge.target_id if edge.source_id==node.id else edge.source_id
            if neighbor_id in seen: continue
            neighbor=g.get(neighbor_id)
            if not neighbor: continue
            # Skip noise: only follow edges to cognitive/strategic content
            if neighbor.type in(IDENTITY,DRIVE,CONVERSATION,META): continue
            # Only follow edges with reasonable confidence
            if edge.confidence<0.4: continue
            expanded.append(neighbor)
            seen.add(neighbor.id)
            connections.append({
                "from":node.content[:40],"from_type":node.type,
                "to":neighbor.content[:40],"to_type":neighbor.type,
                "edge":edge.label or "linked","confidence":round(edge.confidence,2)
            })
        if len(expanded)>=max_total: break

    # Second hop: follow edges from first-hop nodes too (weaker threshold)
    if len(expanded)<max_total:
        first_hop=[n for n in expanded if n.id not in {s.id for s in selected_nodes}]
        for node in first_hop:
            if len(expanded)>=max_total: break
            for edge in g.edges_of(node.id):
                if len(expanded)>=max_total: break
                neighbor_id=edge.target_id if edge.source_id==node.id else edge.source_id
                if neighbor_id in seen: continue
                neighbor=g.get(neighbor_id)
                if not neighbor or neighbor.type in(IDENTITY,DRIVE,CONVERSATION,META): continue
                if edge.confidence<0.6: continue  # higher bar for 2nd hop
                expanded.append(neighbor)
                seen.add(neighbor.id)
                connections.append({
                    "from":node.content[:40],"from_type":node.type,
                    "to":neighbor.content[:40],"to_type":neighbor.type,
                    "edge":edge.label or "linked","confidence":round(edge.confidence,2),
                    "hop":2
                })

    return expanded,connections

# (pipeline_tool_gap removed — tool creation is now inline in the move prompt)

# ═════════════════════════════════════════════════════════════════════════════
#  MOVE GENERATION v7 — single brain pipeline
# ═════════════════════════════════════════════════════════════════════════════
def generate_move(color,game_obj=None,dkg_override=None,is_probe=False):
    """Full pipeline: heat → tools → lens → knowledge → graph expand → move."""
    gm=game_obj or game
    g=dkg_override or players[color]
    cn="Black" if color==BLACK else "White"
    legal=gm.legal_moves(color)
    legal_str=", ".join(f"({r},{c})" for r,c in legal if r>=0)
    if len(legal)<=1: legal_str="pass only"
    b_score,w_score=gm.score()
    board_str=gm.board_string()

    pipeline={"tool_select_time":0,"knowledge_select_time":0,"move_time":0,
              "tool_gap_time":0,"tools_used":[],"knowledge_selected":[],"llm_calls":0,
              "prompt_chars":0,"total_time":0,"new_tool":None,"lens":None,
              "lens_time":0,"score_before":{"black":b_score,"white":w_score},
              "knowledge_node_ids":[],"heat_time":0,"heat_anchors":[],
              "heat_analysis":""}
    t_total=time.time()

    # ── STEP 0: Heat map + position analysis (perceptual layer) ──────
    heat_analysis_text="";heat_anchors=[];heat_map={};heat_components={}
    if CFG.get("enable_heat_analysis",True) and color==WHITE:
        t_heat=time.time()
        try:
            heat_analysis_text,heat_anchors,heat_map,heat_components=run_heat_analysis(g,gm,color)
            pipeline["heat_time"]=round(time.time()-t_heat,1)
            pipeline["heat_anchors"]=[{"pt":list(pt),"h":round(h,2)} for pt,h in heat_anchors[:5]]
            pipeline["heat_analysis"]=heat_analysis_text[:300]
            if heat_analysis_text: pipeline["llm_calls"]+=1
        except Exception as e:
            print(f"  ⚠ heat analysis: {e}")
            pipeline["heat_time"]=round(time.time()-t_heat,1)

    # ── STEP 1: Tool selection + execution ────────────────────────
    tool_outputs,tool_time,tool_err=pipeline_tool_select(g,gm,color)
    pipeline["tool_select_time"]=tool_time
    pipeline["tools_used"]=[o["tool"] for o in tool_outputs]
    pipeline["tool_outputs"]=tool_outputs
    if tool_time>0: pipeline["llm_calls"]+=1

    # ── STEP 1b: Lens selection ───────────────────────────────────
    active_lens,lens_time=pipeline_lens_select(g,gm,color,tool_outputs)
    pipeline["lens"]=active_lens
    pipeline["lens_time"]=lens_time
    if lens_time>0: pipeline["llm_calls"]+=1

    # ── STEP 2: Knowledge selection (lens-weighted) ───────────────
    selected_knowledge,know_time,know_err=pipeline_knowledge_select(g,gm,color,tool_outputs,active_lens=active_lens)
    pipeline["knowledge_select_time"]=know_time
    pipeline["knowledge_selected"]=[n.content[:60] for n in selected_knowledge]
    pipeline["knowledge_node_ids"]=[n.id for n in selected_knowledge]
    # Count passes based on node count
    n_cog=len(g.get_cognitive_nodes(n=100))
    if n_cog>0:
        if n_cog<=CFG["funnel_1pass_max"]: pipeline["llm_calls"]+=1
        elif n_cog<=CFG["funnel_2pass_max"]: pipeline["llm_calls"]+=2
        else: pipeline["llm_calls"]+=3

    # ── STEP 2b: Graph expansion — follow edges from selected nodes ──
    graph_connections=[]
    if selected_knowledge:
        expanded,graph_connections=expand_selection(g,selected_knowledge,max_total=8)
        n_original=len(selected_knowledge)
        selected_knowledge=expanded
        pipeline["graph_expanded"]=len(expanded)-n_original
        pipeline["graph_connections"]=[f"{c['from_type']}→{c['to_type']}({c['edge']})" for c in graph_connections[:5]]
        if graph_connections:
            print(f"    📊 Graph: {n_original} selected → {len(expanded)} after expansion ({len(graph_connections)} edges followed)")

    # ── STEP 3: Move generation ───────────────────────────────────
    # Build rich context from pipeline outputs
    tool_ctx=""
    if tool_outputs:
        tool_ctx="\n\nANALYSIS RESULTS:\n"+"\n".join(
            f"  {o['tool']}: {o['result']}" for o in tool_outputs if o["success"])

    knowledge_ctx=""
    if selected_knowledge:
        # Separate primary selections from graph-expanded nodes
        primary=selected_knowledge[:CFG["max_knowledge_nodes"]]
        connected=selected_knowledge[CFG["max_knowledge_nodes"]:]
        knowledge_ctx="\nYOUR KNOWLEDGE:\n"+"\n".join(
            f"- [{n.type}] {n.content}" for n in primary)
        if connected:
            knowledge_ctx+="\n\nRELATED (from your knowledge graph):\n"+"\n".join(
                f"- [{n.type}] {n.content[:120]}" for n in connected)

    recent=gm.history[-6:]
    hist_str=("\nLAST MOVES: "+", ".join(
        f"{'B' if h[0]==BLACK else 'W'}:{h[1] if h[1]=='pass' else f'({h[1]},{h[2]})'}"
        for h in recent)) if recent else ""

    # ── Working strategy progress block ──
    ws_block=""
    ws_directive=""
    if CFG.get("enable_working_strategy",True) and color==WHITE:
        ws=active_working_strategy(g)
        if ws:
            ws_block="\n"+format_working_strategy_for_prompt(ws,gm)+"\n"
            ws_directive="\nYou may CONTINUE the current plan, or declare a NEW plan with:\n[STRATEGY:N] your plan for the next N moves\n(If you continue, no declaration needed. If the plan is failing, declare new.)"
            pipeline["active_working_strategy"]=ws.content[:80]
        else:
            ws_directive="\nYou have no active plan. DECLARE one for the next several moves:\n[STRATEGY:N] your plan for the next N moves\n(N should be between 4 and 10.)"
            pipeline["active_working_strategy"]=None

    # ── Heat summary block (compact — bulk is in the earlier analysis) ──
    heat_block=""
    if color==WHITE and (heat_analysis_text or heat_anchors):
        hb=format_heat_summary_for_move_prompt(heat_analysis_text,heat_anchors,heat_map,top_k=5)
        if hb: heat_block="\n"+hb+"\n"

    prompt=f"""You are White (O). Move {gm.move_count+1}. {CFG['board_size']}x{CFG['board_size']} Go. (X=Black, O=White you)
{board_str}
Score: B={b_score:.1f} W={w_score:.1f} | Cap: B:{gm.captures[BLACK]} W:{gm.captures[WHITE]}
{hist_str}{heat_block}{ws_block}{tool_ctx}{knowledge_ctx}
Legal: {legal_str}
Black plays {CFG.get('opponent_type','bare_llm').replace('_',' ')}: {('captures, defends atari, reduces opponent liberties, uses influence scoring with star points' if CFG.get('opponent_type')=='heuristic' else 'a bare LLM with no memory, tools, or knowledge — just sees the board and picks moves' if CFG.get('opponent_type','bare_llm')=='bare_llm' else 'random legal moves')}.
{ws_directive}
If you need analysis you don't have, create a tool:
[SAVE_TOOL:name|description]
code (use board, size, inputs variables)
[/SAVE_TOOL]
End with MOVE:row,col or MOVE:pass"""

    sys_msg=f"You are White (O) playing Go. {g.identity_context()} Analyze then choose. End with MOVE:row,col"

    
    messages=[{"role":"system","content":sys_msg},{"role":"user","content":prompt}]
    pipeline["prompt_chars"]=len(prompt)+len(sys_msg)
    pipeline["llm_calls"]+=1
    g.metabolism.spend(g.config.get("llm_call_cost"))

    t0=time.time()
    reply,error=ollama_client._call(messages,timeout=CFG["timeout_move"])
    pipeline["move_time"]=round(time.time()-t0,1)
    elapsed=pipeline["move_time"]

    is_timeout=error and ("timeout" in str(error).lower() or "timed out" in str(error).lower() or elapsed>=CFG["timeout_move"]-5)

    if error or not reply:
        cog_for_fallback=selected_knowledge or g.get_cognitive_nodes(n=5)
        (r,c),method=kg_fallback_move(legal,cog_for_fallback,gm)
        pipeline["total_time"]=round(time.time()-t_total,1)
        detail={"parse":method,"elapsed":elapsed,"knowledge":len(selected_knowledge),
                "confidence":0,"timeout":is_timeout,"error":str(error)[:100] if error else None,
                "fallback":True,"pipeline":pipeline}
        return r,c,f"[error:{error}] {method}",[],[],detail

    # ── Code execution from reply ─────────────────────────────────
    code_outputs=[];new_tools=[]
    actions=ollama_client.extract_actions(reply)
    for action in actions[:2]:
        if action["type"]=="code":
            code=f"board={json.dumps(gm.board)}\nsize={gm.size}\n"+action["code"]
            r=g.run_code(code)
            if r["success"]: code_outputs.append(r["output"][:300])
            else: code_outputs.append(f"err:{r['error'][:100]}")
        elif action["type"]=="save_tool":
            test_code=f"board={json.dumps(gm.board)}\nsize={gm.size}\n"+action["code"]
            r=g.run_code(test_code)
            if r["success"]:
                g.create_tool(action["name"],action["description"],action["code"])
                new_tools.append(action["name"])

    # ── Parse [STRATEGY:N] declaration from reply ─────────────────
    if CFG.get("enable_working_strategy",True) and color==WHITE:
        sm=re.search(r'\[STRATEGY:(\d+)\]\s*([^\n\[]+)',reply,re.IGNORECASE)
        if sm:
            horizon=max(3,min(15,int(sm.group(1))))
            strat_content=sm.group(2).strip()[:300]
            if len(strat_content)>10:
                try:
                    ws=declare_working_strategy(g,strat_content,horizon,gm,color)
                    pipeline["new_working_strategy"]=strat_content[:80]
                    print(f"  📋 NEW PLAN (h={horizon}): {strat_content[:80]}")
                except Exception as e:
                    print(f"  ⚠ WS declare failed: {e}")

    # ── Parse move ────────────────────────────────────────────────
    move_match=re.search(r'MOVE:\s*(\d+)\s*,\s*(\d+)',reply,re.IGNORECASE)
    pass_match=re.search(r'MOVE:\s*pass',reply,re.IGNORECASE)
    pipeline["total_time"]=round(time.time()-t_total,1)
    detail={"parse":"","elapsed":elapsed,"knowledge":len(selected_knowledge),
            "confidence":0,"reply":reply[:200],"timeout":False,"error":None,
            "fallback":False,"pipeline":pipeline}

    if pass_match:
        # ── PASS OVERRIDE: don't allow pass before min_moves ──
        if gm.move_count<CFG["min_moves_before_pass"]:
            cog_for_fallback=selected_knowledge or g.get_cognitive_nodes(n=5)
            (r,c),method=kg_fallback_move(legal,cog_for_fallback,gm)
            detail["parse"]="pass_override→"+method
            return r,c,reply[:200]+f" [pass overridden→{method}]",code_outputs,new_tools,detail
        detail["parse"]="pass"
        return -1,-1,reply[:300],code_outputs,new_tools,detail
    elif move_match:
        detail["parse"]="MOVE:r,c"
        return int(move_match.group(1)),int(move_match.group(2)),reply[:300],code_outputs,new_tools,detail
    else:
        coord=re.search(r'(\d+)\s*,\s*(\d+)',reply)
        if coord:
            detail["parse"]="regex"
            return int(coord.group(1)),int(coord.group(2)),reply[:300],code_outputs,new_tools,detail

        # ── CORRECTION RETRY ──────────────────────────────────
        legal_short=", ".join(f"({r},{c})" for r,c in legal if r>=0)
        correction=messages+[
            {"role":"assistant","content":reply},
            {"role":"user","content":f"You did not include a move. Reply with ONLY:\nMOVE:row,col\n\nLegal: {legal_short}"}
        ]
        pipeline["llm_calls"]+=1
        retry_reply,retry_err=ollama_client._call(correction,timeout=CFG["timeout_correction"])
        if retry_reply and not retry_err:
            rm=re.search(r'MOVE:\s*(\d+)\s*,\s*(\d+)',retry_reply,re.IGNORECASE)
            if rm:
                detail["parse"]="retry_MOVE"
                pipeline["total_time"]=round(time.time()-t_total,1)
                return int(rm.group(1)),int(rm.group(2)),reply[:200]+" [retry]",code_outputs,new_tools,detail
            rc=re.search(r'(\d+)\s*,\s*(\d+)',retry_reply)
            if rc:
                detail["parse"]="retry_regex"
                pipeline["total_time"]=round(time.time()-t_total,1)
                return int(rc.group(1)),int(rc.group(2)),reply[:200]+" [retry]",code_outputs,new_tools,detail

        cog_for_fallback=selected_knowledge or g.get_cognitive_nodes(n=5)
        (r,c),method=kg_fallback_move(legal,cog_for_fallback,gm)
        detail["parse"]=method;detail["fallback"]=True
        pipeline["total_time"]=round(time.time()-t_total,1)
        return r,c,reply[:200]+f" [{method}]",code_outputs,new_tools,detail

# ═════════════════════════════════════════════════════════════════════════════
#  EXECUTE TURN — with probe safety
# ═════════════════════════════════════════════════════════════════════════════
current_game_moves=[]

def execute_turn(game_obj=None,players_map=None,metrics=None,is_probe=False):
    """Execute one turn. Black = configured opponent. White = brain (full pipeline).
    is_probe=True prevents DKG mutation."""
    global current_game_moves
    gm=game_obj or game
    pm=players_map or players
    met=metrics or game_metrics
    if gm.game_over: return{"status":"game_over"}
    color=gm.current;cn="black" if color==BLACK else "white"

    # ── BLACK: Opponent (heuristic / bare_llm / random, per config) ──
    if color==BLACK:
        opponent_player,opp_name=_get_opponent()
        t0=time.time()
        (row,col),reason,method=opponent_player.choose_move(gm,color)
        elapsed=round(time.time()-t0,1)
        success,msg=gm.play(color,row,col)
        if not success:
            gm.play(color,-1,-1);row,col=-1,-1;msg="forced pass"
        move_str="pass" if row==-1 else f"({row},{col})"
        b,w=gm.score()
        entry={"move_num":gm.move_count,"color":cn,"move":move_str,"msg":msg,
               "analysis":reason,"code_outputs":[],"new_tools":[],
               "score":{"black":b,"white":w},"parse":method,"elapsed":elapsed,
               "knowledge":0,"timeout":False,"fallback":False,"error":None,
               "pipeline":{"opponent":opp_name},"score_delta":0,"lens":None}
        met["moves_log"].append(entry)
        if not is_probe:
            game_log.append(entry)
            current_game_moves.append(entry)
        phase_tag=round_state.get("phase_label","?")
        print(f"  [{phase_tag}] B{gm.move_count}: {move_str} [{method}] {elapsed}s")
        return entry

    # ── WHITE: Brain (full pipeline) ──────────────────────────────
    g=pm[WHITE]
    row,col,analysis,code_outputs,new_tools,detail=generate_move(color,game_obj=gm,dkg_override=g,is_probe=is_probe)

    # ── Capture metrics ──
    pip=detail.get("pipeline",{})
    met["latencies"].append(detail.get("elapsed",0))
    met["knowledge_used"].append(detail.get("knowledge",0))
    parse_m=detail.get("parse","unknown")
    met["parse_methods"][parse_m]=met["parse_methods"].get(parse_m,0)+1
    if detail.get("timeout"):
        met["timeouts"]+=1
        met["timeout_details"].append({"move":gm.move_count+1,"color":cn,
            "elapsed":detail.get("elapsed",0),"error":detail.get("error",""),
            "prompt_chars":pip.get("prompt_chars",0),"llm_calls":pip.get("llm_calls",0)})
    if detail.get("fallback"): met["fallbacks"]+=1
    if "retry" in parse_m: met["retries"]+=1
    if detail.get("error") and not detail.get("timeout"): met["parse_failures"]+=1
    met["pipeline_steps"].append(pip)

    success,msg=gm.play(color,row,col);retries=0
    while not success and retries<3:
        legal=gm.legal_moves(color);r,c=random.choice([m for m in legal if m[0]>=0] or[(-1,-1)])
        success,msg=gm.play(color,r,c);row,col=r,c;retries+=1
        detail["parse"]+=f"+retry{retries}"
    if not success:
        if gm.move_count<CFG["min_moves_before_pass"]:
            (row,col),_,_=_heuristic.choose_move(gm,color)
            success,msg=gm.play(color,row,col)
            if not success: gm.play(color,-1,-1);row,col=-1,-1;msg="forced pass"
            else: msg="heuristic(forced)"
        else:
            gm.play(color,-1,-1);row,col=-1,-1;msg="forced pass"
    move_str="pass" if row==-1 else f"({row},{col})"
    b,w=gm.score()

    # ── Score delta for tool evaluation ──
    sb=pip.get("score_before",{})
    my_before=sb.get("white",0)
    my_after=w
    score_delta=round(my_after-my_before,1)
    pip["score_delta"]=score_delta

    # ── Track tool effectiveness ──
    if CFG.get("enable_tool_eval") and pip.get("tools_used"):
        if "tool_deltas" not in met: met["tool_deltas"]={}
        for tool_name in pip["tools_used"]:
            if tool_name not in met["tool_deltas"]: met["tool_deltas"][tool_name]=[]
            met["tool_deltas"][tool_name].append(score_delta)

    # ── Append this move to active working strategy (White only, non-probe) ──
    if not is_probe and CFG.get("enable_working_strategy",True):
        ws=active_working_strategy(g)
        if ws:
            resolved=update_working_strategy(ws,gm,(row,col))
            if resolved:
                pip["resolved_working_strategy"]=resolved.content[:80]
                outcome=resolved.meta.get("outcome","?")
                delta=resolved.meta.get("margin_delta",0)
                print(f"  📋 PLAN RESOLVED [{outcome}]: Δ={delta:+.1f} — {resolved.content[:70]}")

    entry={"move_num":gm.move_count,"color":cn,"move":move_str,"msg":msg,
           "analysis":analysis[:200] if analysis else "","code_outputs":code_outputs,
           "new_tools":new_tools,"score":{"black":b,"white":w},
           "parse":detail.get("parse",""),"elapsed":detail.get("elapsed",0),
           "knowledge":detail.get("knowledge",0),
           "timeout":detail.get("timeout",False),"fallback":detail.get("fallback",False),
           "error":detail.get("error"),
           "pipeline":pip,"score_delta":score_delta,"lens":pip.get("lens")}
    met["moves_log"].append(entry)

    # For training games ONLY, update DKG
    if not is_probe:
        game_log.append(entry)
        current_game_moves.append(entry)
        g.store_conversation(f"As White, move {gm.move_count}: played {move_str}",
            f"Board:\n{gm.board_string()}\nScore: B={b:.1f} W={w:.1f}")
        if code_outputs: g.store_insight(f"Move {gm.move_count} analysis: {' '.join(code_outputs)[:150]}")

        # ── Knowledge outcome tracking: boost/penalize selected nodes ──
        if CFG.get("enable_knowledge_decay") and pip.get("knowledge_node_ids"):
            for nid in pip["knowledge_node_ids"]:
                node=g.get(nid)
                if node:
                    if score_delta>0:
                        node.relevance_score=min(1.0,node.relevance_score+CFG.get("knowledge_win_boost",0.08))
                        if pip.get("lens"):
                            node.router[pip["lens"]]=node.router.get(pip["lens"],0)+0.1
                    elif score_delta<0:
                        node.relevance_score=max(0.01,node.relevance_score-CFG.get("knowledge_lose_penalty",0.05))

        # ── Replay value tracking for tools used this move ──
        if pip.get("tools_used"):
            for tool_name in pip["tools_used"]:
                for t in g.get_tools():
                    tname=t.keywords[0] if t.keywords else t.id
                    if tname==tool_name:
                        effectiveness=1.0 if score_delta>0 else 0.5 if score_delta==0 else 0.0
                        old_rv=t.replay_value
                        t.replay_value=0.8*old_rv+0.2*effectiveness
                        break

        g.tick();g.save()

    phase_tag=round_state.get("phase_label","?")
    to_mark=" ⏱TO" if detail.get("timeout") else ""
    fb_mark=" ↩FB" if detail.get("fallback") else ""
    tools_str=f" 🔬{pip.get('tools_used','')}" if pip.get("tools_used") else ""
    know_str=f" 📚{detail.get('knowledge',0)}" if detail.get("knowledge",0) else ""
    lens_str=f" 🔎{pip.get('lens','')}" if pip.get("lens") else ""
    delta_str=f" Δ{score_delta:+.1f}" if score_delta!=0 else ""
    print(f"  [{phase_tag}] W{gm.move_count}: {move_str} [{detail.get('parse','')}] t:{pip.get('tool_select_time',0)}s l:{pip.get('lens_time',0)}s k:{pip.get('knowledge_select_time',0)}s m:{pip.get('move_time',0)}s #{pip.get('llm_calls',0)}c{lens_str}{tools_str}{know_str}{delta_str}{to_mark}{fb_mark}")
    return entry

# ═════════════════════════════════════════════════════════════════════════════
#  MID-GAME SYNTHESIS — runs every N moves during training (from v5)
# ═════════════════════════════════════════════════════════════════════════════
def run_synthesis():
    """Analyze recent game positions and extract insights/strategies mid-game. Brain plays White."""
    g=players[WHITE]
    convs=g.get_conversations(12)
    if len(convs)<3:
        print(f"  [synth] skip: only {len(convs)} convs")
        return[]
    recent_games=[]
    for c in convs[:8]: recent_games.append(c.content)
    existing=g.get_cognitive_nodes(n=12)
    existing_text="\n".join(f"- [{n.type}] {n.content}" for n in existing)
    connections=[]
    for cog_node in existing[:6]:
        linked=g.edges_of(cog_node.id)
        if linked:
            sources=[]
            for edge in linked:
                tid=edge.target_id if edge.source_id==cog_node.id else edge.source_id
                target=g.get(tid)
                if target and target.type==CONVERSATION: sources.append(target.content[:80])
            if sources: connections.append(f"'{cog_node.content[:60]}' came from: {sources[0]}")
    tools=g.get_tools()
    tool_text="\n".join(f"- {t.keywords[0] if t.keywords else t.id}: {t.content[:80]} (used {t.total_executions}x, rv={t.replay_value:.2f})" for t in tools)
    wm=g.working_memory("Go strategy territory capture liberties attack defend",max_k=6)
    wm_extras=[n for n in wm if n.type not in(CONVERSATION,IDENTITY,DRIVE) and n.id not in {c.id for c in existing}]
    wm_text="\n".join(f"- [{n.type}] {n.content[:100]}" for n in wm_extras) if wm_extras else ""

    opp_desc=CFG.get("opponent_type","bare_llm").replace("_"," ")
    system_msg=f"""You play White on a {CFG['board_size']}x{CFG['board_size']} Go board against a {opp_desc} opponent (Black). Analyze your recent games. Output 1-2 strategic insights.

REQUIRED FORMAT — each insight on its own line with a tag:
[INSIGHT] Your observation about a Go pattern you noticed
[STRATEGY] A specific rule of thumb for making better moves
[THEORY] A general principle about how Go positions work
[GOAL] A priority to hold across future games (only for persistent aims worth carrying forward)

If genuinely nothing new: [NONE]"""

    sections=["RECENT GAME POSITIONS:"]
    for i,t in enumerate(recent_games[:6]): sections.append(f"--- Game position {i+1} ---\n{t}")
    if existing_text: sections.append(f"\nYOUR EXISTING KNOWLEDGE (do NOT repeat these):\n{existing_text}")
    if connections: sections.append(f"\nKNOWLEDGE CONNECTIONS:\n"+"\n".join(connections[:4]))
    if wm_text: sections.append(f"\nRELATED CONCEPTS:\n{wm_text}")
    if tool_text: sections.append(f"\nYOUR ANALYSIS TOOLS:\n{tool_text}")
    sections.append("\nWhat NEW patterns or strategies do you see?")
    prompt="\n".join(sections)
    total_chars=len(system_msg)+len(prompt)
    print(f"  [synth] {len(convs)} convs, {len(existing)} cog, {total_chars} chars")

    g.metabolism.spend(g.config.get("synthesis_cost"))
    reply,error=ollama_client._call([{"role":"system","content":system_msg},{"role":"user","content":prompt}],timeout=CFG["timeout_review"])
    if error or not reply:
        simple=f"You play White on a {CFG['board_size']}x{CFG['board_size']} Go board. Give one [INSIGHT] about Go strategy."
        reply,error=ollama_client._call([{"role":"user","content":simple}],timeout=CFG["timeout_review"])
        if error or not reply: return[]

    if "[NONE]" in reply.upper() and len(reply)<60: return[]
    results=[];type_map={"INSIGHT":"insight","THEORY":"theory","STRATEGY":"strategy","GOAL":"goal"}
    conv_ids=[c.id for c in convs[:3]]

    # Prepare boards for critic birth test
    home_board=_extract_board_from_conv(convs[0]) if convs else None
    exclude_ids={c.id for c in convs[:2]}
    other_board=_get_comparison_board(g,exclude_ids=exclude_ids)
    critic_on=CFG.get("enable_critic",True) and home_board and other_board

    for line in reply.strip().split("\n"):
        line=line.strip()
        if not line or len(line)<15: continue
        for tag,nt in type_map.items():
            if f"[{tag}]" in line.upper():
                idx=line.upper().find(f"[{tag}]")
                content=line[idx+len(tag)+2:].strip()
                for p in["**","*","- ","• ",": "]:
                    if content.startswith(p): content=content[len(p):]
                content=content.strip("*").strip()
                if content and len(content)>10:
                    n=g.store_cognitive(nt,content,conv_ids)
                    # CRITIC BIRTH TEST
                    critic_verdict="skipped"
                    if n is not None and critic_on:
                        outcome,chosen,_,_=critic_test_board_fit(
                            content,home_board,other_board,source="synthesis_birth")
                        if outcome!="error":
                            record_critic_test(n,outcome)
                            critic_verdict=f"{outcome}({chosen})"
                            if outcome in("miss","vague"):
                                n.relevance_score=max(0.1,n.relevance_score*0.4)
                        else:
                            critic_verdict="error"
                    results.append({"type":nt,"content":content,"new":n is not None,"critic":critic_verdict})
                    print(f"  [synth] ✓ {nt}: {content[:50]} [critic:{critic_verdict}]")
                break
    if not results:
        for line in reply.strip().split("\n"):
            line=line.strip()
            if len(line)<25 or line.upper().startswith("[NONE"): continue
            if line.startswith("#") or line.startswith("---"): continue
            content=line
            for p in["**","*","- ","• ","1. ","2. ","– ","— ","> "]:
                if content.startswith(p): content=content[len(p):]
            content=content.strip("*").strip()
            if len(content)>15:
                n=g.store_cognitive("insight",content[:200],conv_ids)
                # CRITIC BIRTH TEST (fallback path)
                critic_verdict="skipped"
                if n is not None and critic_on:
                    outcome,chosen,_,_=critic_test_board_fit(
                        content[:200],home_board,other_board,source="synthesis_birth")
                    if outcome!="error":
                        record_critic_test(n,outcome)
                        critic_verdict=f"{outcome}({chosen})"
                        if outcome in("miss","vague"):
                            n.relevance_score=max(0.1,n.relevance_score*0.4)
                results.append({"type":"insight","content":content[:200],"new":n is not None,"critic":critic_verdict})
                print(f"  [synth] fallback: {content[:50]} [critic:{critic_verdict}]")
                break
    actions=ollama_client.extract_actions(reply)
    for a in actions:
        if a["type"]=="save_tool":
            test=f"board=[[0]*{CFG['board_size']} for _ in range({CFG['board_size']})]\nsize={CFG['board_size']}\n"+a["code"]
            r=g.run_code(test)
            if r["success"]:
                g.create_tool(a["name"],a["description"],a["code"])
                results.append({"type":"tool","content":f"Created tool: {a['name']}"})
    g.save();return results

# ═════════════════════════════════════════════════════════════════════════════
#  POST-GAME REVIEW — brain (White) vs configured opponent (Black), 3-pass
# ═════════════════════════════════════════════════════════════════════════════
def post_game_review(g,game_num,winner,b_score,w_score,move_log):
    """3-pass review of White's play against the opponent. Stores in KG."""
    if not CFG["enable_post_game_review"] or len(move_log)<5:
        return _legacy_synthesis(g)

    won=(winner=="White")
    opp_desc=CFG.get("opponent_type","bare_llm").replace("_"," ")

    # Build game narrative from move log
    score_timeline=[]
    captures_log=[]
    for e in move_log:
        sc=e.get("score",{})
        score_timeline.append(f"M{e['move_num']}: {e['color'][0].upper()}{e['move']} B={sc.get('black',0):.0f} W={sc.get('white',0):.0f}")
        if "cap" in(e.get("msg","")): captures_log.append(f"M{e['move_num']}: {e['color']} captured")

    # Subsample for context window
    if len(score_timeline)>20:
        step=len(score_timeline)//20
        score_timeline=score_timeline[::step]+[score_timeline[-1]]

    narrative="\n".join(score_timeline)
    result_str=f"{'WON' if won else 'LOST'} as White (B={b_score:.0f} W={w_score:.0f}, {len(move_log)} moves)"

    # ── PASS 1: Game review ──────────────────────────────────────
    prompt1=f"""You are White. You just {result_str} against a {opp_desc} opponent (Black).

GAME TIMELINE:
{narrative}

{('CAPTURES: '+chr(10).join(captures_log)) if captures_log else 'No captures.'}

Review this game. Where did you gain or lose ground? What did Black exploit? Be specific about move numbers."""

    print(f"  [review] Pass 1: game narrative...")
    reply1,err1=ollama_client._call([{"role":"user","content":prompt1}],timeout=CFG["timeout_review"])
    if err1 or not reply1:
        print(f"  [review] Pass 1 failed: {err1}")
        return _legacy_synthesis(g)

    # ── PASS 2: Turning points ───────────────────────────────────
    prompt2=f"""Based on your review:
{reply1[:800]}

Identify the 3 most important turning points.
What should you have done differently?"""

    print(f"  [review] Pass 2: turning points...")
    reply2,err2=ollama_client._call([{"role":"user","content":prompt2}],timeout=CFG["timeout_review"])
    if err2 or not reply2:
        print(f"  [review] Pass 2 failed: {err2}")
        return _extract_and_store(g,reply1,"Brain")

    # ── PASS 3: Lessons ──────────────────────────────────────────
    existing=g.get_cognitive_nodes(n=15)
    existing_text="\n".join(f"- [{n.type}] {n.content[:80]}" for n in existing) if existing else "(none)"

    prompt3=f"""Turning points:
{reply2[:800]}

YOUR EXISTING KNOWLEDGE (do NOT repeat these):
{existing_text}

Write 1-3 NEW specific rules for playing White against this opponent.
Format each as:
[STRATEGY] specific rule
or [INSIGHT] specific observation
or [GOAL] a persistent priority worth carrying into future games

Only write NEW knowledge not covered above."""

    print(f"  [review] Pass 3: lessons...")
    reply3,err3=ollama_client._call([{"role":"user","content":prompt3}],timeout=CFG["timeout_review"])
    if err3 or not reply3:
        return _extract_and_store(g,reply2,"Brain")

    return _extract_and_store(g,reply3,"Brain")

def _extract_and_store(g,reply,cn):
    """Parse [INSIGHT]/[STRATEGY]/[THEORY]/[GOAL] tags and store in KG.
    Each new node gets a critic birth test (two-board fit) before admission."""
    results=[]
    type_map={"INSIGHT":"insight","THEORY":"theory","STRATEGY":"strategy","GOAL":"goal"}
    recent_convs=g.get_conversations(5)
    conv_ids=[c.id for c in recent_convs[:3]]
    # Home board: most recent conversation's board (the game just played)
    home_board=_extract_board_from_conv(recent_convs[0]) if recent_convs else None
    # Other board: pick from earlier games (excluding the most recent few)
    exclude_ids={c.id for c in recent_convs[:2]}
    other_board=_get_comparison_board(g,exclude_ids=exclude_ids)

    for line in reply.strip().split("\n"):
        line=line.strip()
        if not line or len(line)<15: continue
        for tag,nt in type_map.items():
            if f"[{tag}]" in line.upper():
                idx=line.upper().find(f"[{tag}]")
                content=line[idx+len(tag)+2:].strip()
                for p in["**","*","- ","• ",": "]:
                    if content.startswith(p): content=content[len(p):]
                content=content.strip("*").strip()
                if content and len(content)>10:
                    # Store first
                    n=g.store_cognitive(nt,content,conv_ids)
                    # Then critic birth test (if we have two boards to compare)
                    critic_verdict="skipped"
                    if n is not None and home_board and other_board and CFG.get("enable_critic",True):
                        outcome,chosen,reasoning,elapsed=critic_test_board_fit(
                            content,home_board,other_board,source="review_birth")
                        if outcome!="error":
                            record_critic_test(n,outcome)
                            critic_verdict=f"{outcome}({chosen})"
                            # Hammer relevance if failed or vague at birth
                            if outcome in("miss","vague"):
                                n.relevance_score=max(0.1,n.relevance_score*0.4)
                        else:
                            critic_verdict="error"
                    results.append({"type":nt,"content":content,"new":n is not None,"critic":critic_verdict})
                    print(f"  [review] ✓ {nt}: {content[:50]} [critic:{critic_verdict}]")
                break
    if not results:
        for line in reply.strip().split("\n"):
            line=line.strip()
            if len(line)>25 and not line.upper().startswith("[NONE"):
                content=line[:200]
                for p in["**","*","- ","• ","1. ","2. "]:
                    if content.startswith(p): content=content[len(p):]
                n=g.store_cognitive("insight",content.strip(),conv_ids)
                results.append({"type":"insight","content":content.strip(),"new":n is not None})
                break
    g.save()
    return results

def _legacy_synthesis(g):
    """Minimal fallback if review fails — quick synthesis."""
    convs=g.get_conversations(8)
    if len(convs)<2: return[]
    conv_text="\n".join(c.content[:150] for c in convs[:5])
    opp_desc=CFG.get("opponent_type","bare_llm").replace("_"," ")
    prompt=f"You play White on a {CFG['board_size']}x{CFG['board_size']} Go board against a {opp_desc} opponent. Recent game positions:\n{conv_text}\n\nGive one [INSIGHT] or [STRATEGY] about Go."
    reply,err=ollama_client._call([{"role":"user","content":prompt}],timeout=CFG["timeout_review"])
    if err or not reply: return[]
    return _extract_and_store(g,reply,"Brain")

# ═════════════════════════════════════════════════════════════════════════════
#  DREAM TOOL REVIEW — numeric replay-value pass, no LLM call
# ═════════════════════════════════════════════════════════════════════════════
def dream_tool_review(g):
    """Fast numeric review of tool replay value. Hit hard if low, boost if high.
    No LLM call. Tools survive by proving themselves in play."""
    tools=g.get_tools()
    if not tools: return[]
    actions=[]
    now=time.time()
    for t in tools:
        tname=t.keywords[0] if t.keywords else t.id
        rv=t.replay_value
        # Recency penalty: decay replay_value if tool hasn't been used recently
        hours_since=max(0,(now-t.last_executed)/3600.0) if t.last_executed else 999
        if hours_since>2 and t.total_executions>0:
            # Hasn't been used in 2+ hours — pull replay_value toward 0
            recency_decay=min(0.15, hours_since*0.02)
            t.replay_value=max(0.0, rv-recency_decay)
            rv=t.replay_value
        # Demand: if many moves played but tool rarely selected, penalize
        if t.total_executions==0 and t.age_hours()>1:
            t.replay_value=max(0.0, rv*0.7)
            rv=t.replay_value

        old_rel=t.relevance_score
        if rv<0.15:
            # Critical — big hit
            t.relevance_score=max(0.01, t.relevance_score*0.3)
            actions.append(f"⚠ tool '{tname}' hammered: rv={rv:.2f} rel {old_rel:.2f}→{t.relevance_score:.2f}")
        elif rv<0.3:
            # Low — significant hit
            t.relevance_score=max(0.01, t.relevance_score*0.5)
            actions.append(f"↓ tool '{tname}' hit: rv={rv:.2f} rel {old_rel:.2f}→{t.relevance_score:.2f}")
        elif rv<0.45:
            # Mediocre — moderate decay
            t.relevance_score=max(0.01, t.relevance_score*0.8)
        elif rv>0.7:
            # Strong — boost
            t.relevance_score=min(1.0, t.relevance_score*1.15+0.02)
            actions.append(f"↑ tool '{tname}' boosted: rv={rv:.2f} rel {old_rel:.2f}→{t.relevance_score:.2f}")
        # else: neutral range (0.45-0.7), leave alone
    return actions

# ═════════════════════════════════════════════════════════════════════════════
#  DREAM CRITIC SAMPLE — re-test N existing cognitive nodes per dream cycle
# ═════════════════════════════════════════════════════════════════════════════
def dream_critic_sample(g,sample_size=None):
    """Sample N cognitive nodes, run a two-board fit test on each,
    update critic_score. Nodes with persistently low critic_score get hammered."""
    if not CFG.get("enable_critic",True): return []
    sample_size=sample_size or CFG.get("critic_dream_sample",3)
    cog=g.get_cognitive_nodes(n=50)
    if len(cog)<2: return []
    # Prioritize high-relevance, low-critic-score nodes for re-testing
    # (suspicious combination: brain trusts it a lot, critic may not)
    cog.sort(key=lambda n: n.relevance_score - n.critic_score*0.5, reverse=True)
    sampled=cog[:sample_size]

    actions=[]
    for n in sampled:
        # Pick a home board: a conversation the node is linked to, or most recent
        linked_conv=None
        for edge in g.edges_of(n.id):
            other_id=edge.target_id if edge.source_id==n.id else edge.source_id
            other=g.get(other_id)
            if other and other.type==CONVERSATION:
                linked_conv=other;break
        if not linked_conv:
            convs=g.get_conversations(5)
            if not convs: continue
            import random as _r
            linked_conv=_r.choice(convs)
        home_board=_extract_board_from_conv(linked_conv)
        if not home_board: continue
        # Other board from a different conversation
        other_board=_get_comparison_board(g,exclude_ids={linked_conv.id})
        if not other_board: continue

        outcome,chosen,_,elapsed=critic_test_board_fit(
            n.content,home_board,other_board,source="dream_sample")
        if outcome=="error": continue
        record_critic_test(n,outcome)
        passed=(outcome=="match")
        # Hammer nodes whose critic_score is falling after multiple tests
        if len(n.critic_history)>=3 and n.critic_score<0.35:
            old_rel=n.relevance_score
            n.relevance_score=max(0.05, n.relevance_score*0.5)
            actions.append(f"⚠ critic hammered '{n.content[:40]}…' cs={n.critic_score:.2f} rel {old_rel:.2f}→{n.relevance_score:.2f}")
        elif passed and n.critic_score>0.7 and n.relevance_score<0.5:
            n.relevance_score=min(1.0, n.relevance_score*1.1+0.02)
            actions.append(f"↑ critic rescued '{n.content[:40]}…' cs={n.critic_score:.2f}")
    return actions

# ═════════════════════════════════════════════════════════════════════════════
#  WS DISTILLATION — turn resolved working strategies into insights
# ═════════════════════════════════════════════════════════════════════════════
def distill_resolved_working_strategies(g):
    """For each resolved WS in this game, distill outcome into INSIGHT/STRATEGY.
    Succeeded plans become positive patterns; failed plans become cautions.
    After distillation the WS is archived."""
    if not CFG.get("enable_working_strategy",True): return
    resolved=[n for n in g.nodes_of_type(WORKING_STRATEGY)
              if n.meta.get("state")=="resolved" and not n.meta.get("distilled")]
    if not resolved: return
    # Limit distillation count per cycle to avoid explosion
    resolved.sort(key=lambda n: n.meta.get("resolved_at_move",0))
    distilled_count=0
    for ws in resolved[:5]:  # at most 5 per game
        outcome=ws.meta.get("outcome","neutral")
        delta=ws.meta.get("margin_delta",0)
        declared_score=ws.meta.get("declared_score",{})
        resolved_score=ws.meta.get("resolved_score",{})
        moves=ws.meta.get("moves_under_plan",[])
        moves_str=", ".join(f"({m['move'][0]},{m['move'][1]})" for m in moves[:8]
                            if isinstance(m.get("move"),(tuple,list)) and m["move"][0]>=0)

        # Only distill plans that actually ran (some moves) and have meaningful outcome
        if len(moves)<2: continue

        if outcome=="succeeded":
            content=f"Plan '{ws.content[:120]}' worked: executed over {len(moves)} moves, margin improved by {delta:+.1f}. Moves: {moves_str}"
            ntype="strategy"
        elif outcome=="failed":
            content=f"Plan '{ws.content[:120]}' failed: executed over {len(moves)} moves, margin worsened by {delta:+.1f}. Moves: {moves_str}"
            ntype="insight"  # cautionary insight
        else:
            # neutral outcome — archive without distilling
            ws.meta["distilled"]=True
            continue

        recent_convs=g.get_conversations(3)
        source_ids=[c.id for c in recent_convs[:2]]+[ws.id]
        n=g.store_cognitive(ntype,content,source_ids,confidence=0.6 if outcome=="succeeded" else 0.5)
        ws.meta["distilled"]=True
        if n:
            # Link the WS to the new insight
            g.add_edge(Edge(ws.id,n.id,channels={"distilled_to":0.9},
                confidence=0.8,label="distilled"))
            # Carry WS critic_score to the new node as a prior
            if ws.critic_history:
                n.critic_history=list(ws.critic_history)
                n.critic_score=ws.critic_score
            distilled_count+=1
            tag="✓" if outcome=="succeeded" else "✗"
            print(f"  📋→📘 distilled {tag}: {content[:80]}")

        # Archive the WS after distillation (mark relevance low so it ages out)
        ws.relevance_score=0.1
        ws.meta["state"]="distilled"
    if distilled_count:
        g.save()

# ═════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE DECAY — penalize unused nodes after each game
# ═════════════════════════════════════════════════════════════════════════════
def apply_knowledge_decay():
    """After each game, decay nodes that were never selected. Single brain."""
    if not CFG.get("enable_knowledge_decay"): return
    g=players[WHITE]  # brain plays White
    cog=g.get_cognitive_nodes(n=100)
    # Collect all node IDs that were used this game
    used_ids=set()
    for entry in game_metrics.get("moves_log",[]):
        pip=entry.get("pipeline",{})
        for nid in pip.get("knowledge_node_ids",[]):
            used_ids.add(nid)
    # Decay unused cognitive nodes
    decay_amt=CFG.get("knowledge_unused_decay",0.02)
    decayed=0
    for n in cog:
        if n.id not in used_ids:
            n.relevance_score=max(0.01,n.relevance_score-decay_amt)
            decayed+=1
    if decayed>0:
        print(f"  📉 Brain: decayed {decayed} unused knowledge nodes by {decay_amt}")
    g.save()

# ═════════════════════════════════════════════════════════════════════════════
#  LENS ROUTER UPDATE — strengthen lens-node associations after game outcome
# ═════════════════════════════════════════════════════════════════════════════
def update_lens_routers(won_color):
    """After a game, boost lens→node associations based on move outcomes. Single brain, all moves."""
    g=players[WHITE]  # brain plays White
    for entry in game_metrics.get("moves_log",[]):
        pip=entry.get("pipeline",{})
        lens=pip.get("lens")
        delta=entry.get("score_delta",0)
        if not lens: continue
        for nid in pip.get("knowledge_node_ids",[]):
            node=g.get(nid)
            if node:
                # Boost lens association for positive outcomes, decay for negative
                current=node.router.get(lens,0)
                if delta>0: node.router[lens]=min(1.0,current+0.05)
                elif delta<-2: node.router[lens]=max(-0.5,current-0.03)
    g.save()

# (Control baseline removed — fixed opponent IS the baseline)

# ═════════════════════════════════════════════════════════════════════════════
#  SNAPSHOTS
# ═════════════════════════════════════════════════════════════════════════════
def _save_snap(color_name,label):
    g=players[WHITE]  # brain plays White
    path=os.path.join(SNAP_DIR,f"{color_name}_{label}.json")
    old=g.SAVE_PATH;g.SAVE_PATH=path;g.save();g.SAVE_PATH=old
    meta={"label":f"{color_name}_{label}","color":color_name,"game":scoreboard["total_games"],
          "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),"nodes":len(g.nodes),"edges":len(g.edges),
          "cognitive":len(g.get_cognitive_nodes(n=100)),"tools":len(g.get_tools())}
    with open(os.path.join(SNAP_DIR,f"{color_name}_{label}_meta.json"),"w") as f: json.dump(meta,f,indent=2)
    print(f"  📸 {color_name}_{label} ({meta['nodes']}n {meta['cognitive']}c)")

def _load_snap(label):
    path=os.path.join(SNAP_DIR,f"{label}.json")
    if not os.path.exists(path): return None
    d=DKG();d.SAVE_PATH=path;d.load();return d

def _list_snaps():
    out=[]
    for f in sorted(os.listdir(SNAP_DIR)):
        if f.endswith("_meta.json"):
            with open(os.path.join(SNAP_DIR,f)) as fh: out.append(json.load(fh))
    return out

# ═════════════════════════════════════════════════════════════════════════════
#  PROBE GAME — frozen brain (White) vs configured opponent (Black). No DKG mutation.
# ═════════════════════════════════════════════════════════════════════════════
def run_probe_game(white_label):
    """Play a full probe game. Frozen brain as White. Heuristic as Black. No mutation."""
    t_start=time.time()
    white_dkg=_load_snap(white_label)
    if not white_dkg:
        print(f"  ⚠ Probe: missing snapshot {white_label}")
        return None
    gm=GoGame(size=CFG["board_size"],komi=CFG["komi"],move_cap=CFG["move_cap"])
    met=_new_game_metrics()
    probe_players={BLACK:None,WHITE:white_dkg}  # Black=opponent (handled by _get_opponent), White=frozen brain

    # Show probe on GUI
    global game,game_log
    with game_lock:
        game=gm
        game_log.clear()

    while not gm.game_over:
        if time.time()-t_start>CFG["probe_max_time"]:
            met["timeouts"]+=1
            print(f"  ⚠ Probe game timeout ({CFG['probe_max_time']}s)")
            break
        entry=execute_turn(game_obj=gm,players_map=probe_players,metrics=met,is_probe=True)
        if entry and entry.get("status")!="game_over":
            game_log.append(entry)
        time.sleep(auto_speed)

    if not gm.game_over:
        gm.end_game()

    met["total_duration"]=round(time.time()-t_start,1)
    b,w=gm.score()
    winner_name={BLACK:"black",WHITE:"white",None:"draw"}.get(gm.winner,"?")
    margin=round(w-b,1)  # positive = White ahead
    return {
        "winner":winner_name,"white_label":white_label,
        "black_score":b,"white_score":w,"margin":margin,
        "moves":gm.move_count,
        "b_captures":gm.captures[BLACK],"w_captures":gm.captures[WHITE],
        "duration":met["total_duration"],"timeouts":met["timeouts"],
        "fallbacks":met["fallbacks"],"retries":met["retries"],
        "avg_latency":round(sum(met["latencies"])/max(1,len(met["latencies"])),1),
        "parse_methods":met["parse_methods"],"time":time.strftime("%H:%M:%S"),
    }

# ═════════════════════════════════════════════════════════════════════════════
#  BACKGROUND THREAD — 2-game rounds: train + probe
# ═════════════════════════════════════════════════════════════════════════════
games_played=scoreboard["total_games"]

def background_loop():
    global auto_play,game,games_played,scoreboard,eval_scoreboard,current_game_moves
    global game_metrics,round_state,round_history
    last_synth_move=0
    while True:
        time.sleep(1)
        if not auto_play: continue

        # ═══ PHASE 1: TRAINING GAME (brain White vs configured opponent) ════
        if round_state["phase"]=="train":
            if game.game_over:
                games_played+=1;b,w=game.score()
                winner="Black" if game.winner==BLACK else "White" if game.winner==WHITE else "Draw"
                margin=round(w-b,1)
                print(f"\n  ════ ROUND {round_state['round']} — TRAIN COMPLETE: {winner} (B:{b:.1f} W:{w:.1f} margin:{margin:+.1f}) ════")
                if game.winner==BLACK: scoreboard["black_wins"]+=1
                elif game.winner==WHITE: scoreboard["white_wins"]+=1
                else: scoreboard["draws"]+=1
                scoreboard["total_games"]=games_played
                game_metrics["total_duration"]=round(time.time()-game_metrics.get("_start_time",time.time()),1)
                train_result={"game":games_played,"winner":winner,"black_score":b,"white_score":w,
                    "margin":margin,"moves":game.move_count,
                    "b_captures":game.captures[BLACK],"w_captures":game.captures[WHITE],
                    "time":time.strftime("%H:%M:%S"),"phase":"train",
                    "timeouts":game_metrics["timeouts"],"fallbacks":game_metrics["fallbacks"],
                    "retries":game_metrics["retries"],"parse_methods":game_metrics["parse_methods"],
                    "avg_latency":round(sum(game_metrics["latencies"])/max(1,len(game_metrics["latencies"])),1),
                    "duration":game_metrics["total_duration"]}
                scoreboard["games"].append(train_result)
                if len(scoreboard["games"])>100: scoreboard["games"]=scoreboard["games"][-100:]
                _save_sb(scoreboard)
                print(f"  Record: B {scoreboard['black_wins']} — W {scoreboard['white_wins']}")

                game_file=os.path.join(GAMES_DIR,f"game_{games_played:03d}_train.json")
                with open(game_file,"w") as f:
                    json.dump({"game":games_played,"phase":"train","winner":winner,"black_score":b,"white_score":w,
                        "moves":game.move_count,"move_log":game_metrics["moves_log"]},f,indent=1)
                current_game_moves=[]

                # ── POST-GAME: review, dream, decay ──────────────────
                g=players[WHITE]
                g.store_conversation(f"Game {games_played}: {'WON' if winner=='White' else 'LOST'} as White. B={b:.1f} W={w:.1f}",
                    f"Captures: B:{game.captures[BLACK]} W:{game.captures[WHITE]}. Moves: {game.move_count}.")
                g.update_routers([n.id for n in g.get_cognitive_nodes(n=10)],
                    outcome_positive=(winner=="White"))
                # ── Resolve any lingering active working strategies ──
                try:
                    for ws in g.nodes_of_type(WORKING_STRATEGY):
                        if ws.meta.get("state","active")=="active":
                            resolve_working_strategy(ws,game,"game ended")
                            print(f"  📋 PLAN FINAL [{ws.meta.get('outcome','?')}]: Δ={ws.meta.get('margin_delta',0):+.1f} — {ws.content[:60]}")
                except Exception as e: print(f"  ⚠ WS resolve: {e}")
                try:
                    results=post_game_review(g,games_played,winner,b,w,game_metrics["moves_log"])
                    for r in results: print(f"  ✦ Brain [{r['type']}]: {r['content'][:60]}")
                except Exception as e: print(f"  ⚠ Review: {e}");traceback.print_exc()
                # ── Distill resolved working strategies into insights ──
                try:
                    distill_resolved_working_strategies(g)
                except Exception as e: print(f"  ⚠ WS distill: {e}")
                try:
                    acts=g.dream_cycle()
                    if acts: print(f"  💤 Brain: {', '.join(acts[:5])}")
                    tool_acts=dream_tool_review(g)
                    if tool_acts:
                        for ta in tool_acts: print(f"  🔧 {ta}")
                    critic_acts=dream_critic_sample(g)
                    if critic_acts:
                        for ca in critic_acts: print(f"  🧐 {ca}")
                except Exception as e: print(f"  ⚠ Dream: {e}")
                try: apply_knowledge_decay()
                except Exception as e: print(f"  ⚠ Decay: {e}")
                try: update_lens_routers(game.winner)
                except Exception as e: print(f"  ⚠ Lens update: {e}")
                g.save()

                _save_snap("brain",f"g{games_played}")
                round_state["_train_result"]=train_result

                g0_exists=os.path.exists(os.path.join(SNAP_DIR,"brain_g0.json"))
                if not g0_exists or games_played<1:
                    round_state["_probe_result"]=None
                    round_state["phase"]="round_done";continue

                round_state["phase"]="probe";round_state["phase_num"]=2;round_state["phase_label"]="PROBE"
                time.sleep(3)
                with game_lock:
                    game=GoGame(size=CFG["board_size"],komi=CFG["komi"],move_cap=CFG["move_cap"])
                    game_log.clear();game_metrics=_new_game_metrics()
                print(f"\n  ──── ROUND {round_state['round']} — PROBE: Brain(g{games_played}) as White vs Opponent ────\n")
                continue

            # Normal training turn
            with game_lock:
                try: execute_turn()
                except Exception as e: print(f"  ⚠ {e}");traceback.print_exc()
            time.sleep(auto_speed)
            # ── Mid-game synthesis every N moves ──
            if game.move_count-last_synth_move>=CFG["synthesis_interval"]:
                last_synth_move=game.move_count
                try:
                    results=run_synthesis()
                    for r in results: print(f"  ✦ Brain [{r['type']}]: {r['content'][:50]}")
                except Exception as e: print(f"  ⚠ Synthesis: {e}")
            # ── Dream cycle every 20 moves ──
            if game.move_count%20==0 and game.move_count>0:
                g=players[WHITE]
                try:
                    acts=g.dream_cycle()
                    if acts: print(f"  💤 {', '.join(acts[:3])}")
                    tool_acts=dream_tool_review(g)
                    if tool_acts:
                        for ta in tool_acts: print(f"  🔧 {ta}")
                except: pass

        # ═══ PHASE 2: PROBE (frozen brain_gN White vs configured opponent) ═══
        elif round_state["phase"]=="probe":
            print(f"  ── Running PROBE: Brain(g{games_played}) as White vs Opponent ──")
            try:
                probe=run_probe_game(f"brain_g{games_played}")
                if probe:
                    mark="○" if probe["winner"]=="white" else "●" if probe["winner"]=="black" else "="
                    print(f"  {mark} PROBE → {probe['winner']} B:{probe['black_score']:.0f} W:{probe['white_score']:.0f} margin:{probe['margin']:+.1f} ({probe['moves']}mv)")
                    eval_scoreboard["games"].append({"game":games_played,"round":round_state["round"],**probe})
                    pf=os.path.join(GAMES_DIR,f"game_{games_played:03d}_probe.json")
                    with open(pf,"w") as f: json.dump(probe,f,indent=1)
                round_state["_probe_result"]=probe
            except Exception as e:
                print(f"  ⚠ PROBE error: {e}");traceback.print_exc()
                round_state["_probe_result"]=None

            _save_eval_sb(eval_scoreboard)
            round_state["phase"]="round_done"

        # ═══ ROUND COMPLETE ═══════════════════════════════════════════
        elif round_state["phase"]=="round_done":
            train_r=round_state.get("_train_result")
            probe=round_state.get("_probe_result")
            white_won_probe=probe and probe["winner"]=="white"

            # Score trajectory
            probe_games=eval_scoreboard.get("games",[])
            white_wins=sum(1 for e in probe_games if e.get("winner")=="white")
            margins=[e.get("margin",0) for e in probe_games]
            avg_margin=round(sum(margins)/max(1,len(margins)),1)

            round_result={
                "round":round_state["round"],"game":games_played,
                "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
                "train":_safe_result(train_r),
                "probe":_safe_result(probe),
                "white_won_probe":white_won_probe,
                "probe_margin":probe.get("margin",0) if probe else None,
                "cumulative_white_wins":white_wins,
                "cumulative_probes":len(probe_games),
                "avg_margin":avg_margin,
            }
            round_state["current_round_results"]=round_result
            round_history.append(round_result);_save_round_history(round_history)
            rf=os.path.join(ROUNDS_DIR,f"round_{round_state['round']:03d}.json")
            with open(rf,"w") as f: json.dump(round_result,f,indent=2)

            train_margin=train_r.get("margin",0) if train_r else 0
            probe_margin=probe.get("margin",0) if probe else 0
            print(f"\n  ╔══════════════════════════════════════════════════╗")
            print(f"  ║  ROUND {round_state['round']} COMPLETE                              ║")
            print(f"  ╠══════════════════════════════════════════════════╣")
            if train_r: print(f"  ║  TRAIN: {train_r['winner']:>5}  B:{train_r['black_score']:5.1f}  W:{train_r['white_score']:5.1f}  m:{train_margin:+5.1f}  {train_r['moves']:>2}mv ║")
            if probe:   print(f"  ║  PROBE: {probe['winner']:>5}  B:{probe['black_score']:5.1f}  W:{probe['white_score']:5.1f}  m:{probe_margin:+5.1f}  {probe['moves']:>2}mv ║")
            print(f"  ║  Probe wins: {white_wins}/{len(probe_games)}  Avg margin: {avg_margin:+.1f}    ║")
            print(f"  ╚══════════════════════════════════════════════════╝\n")

            time.sleep(3)
            round_state["round"]+=1;round_state["phase"]="train";round_state["phase_num"]=1;round_state["phase_label"]="TRAIN"
            for k in["_train_result","_probe_result"]: round_state.pop(k,None)
            with game_lock:
                game=GoGame(size=CFG["board_size"],komi=CFG["komi"],move_cap=CFG["move_cap"])
                game_log.clear();game_metrics=_new_game_metrics();last_synth_move=0
            critic_reset_game_stats()
            print(f"\n  ════ ROUND {round_state['round']} — TRAIN GAME STARTING ════\n")

        time.sleep(auto_speed if round_state["phase"]=="train" else 0.5)

def _safe_result(r):
    if not r: return None
    return {k:v for k,v in r.items() if k not in("metrics","moves_log")}

bg_thread=threading.Thread(target=background_loop,daemon=True)

# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index(): return render_template("game.html")

@app.route("/api/game")
def api_game():
    d=game.state_dict()
    d["games_played"]=games_played;d["scoreboard"]=scoreboard
    d["round"]=round_state["round"];d["phase"]=round_state["phase"]
    d["phase_num"]=round_state["phase_num"];d["phase_label"]=round_state["phase_label"]
    pips=game_metrics.get("pipeline_steps",[])
    d["metrics"]={
        "timeouts":game_metrics["timeouts"],"fallbacks":game_metrics["fallbacks"],
        "retries":game_metrics["retries"],"parse_failures":game_metrics["parse_failures"],
        "avg_latency":round(sum(game_metrics["latencies"])/max(1,len(game_metrics["latencies"])),1),
        "total_moves":len(game_metrics["moves_log"]),
        "total_calls":sum(p.get("llm_calls",0) for p in pips),
        "avg_prompt_tok":round(sum(p.get("prompt_chars",0) for p in pips)/max(1,len(pips))/4),
    }
    return jsonify(d)

@app.route("/api/game/move",methods=["POST"])
def api_move():
    with game_lock: return jsonify(execute_turn())

@app.route("/api/game/auto",methods=["POST"])
def api_auto():
    global auto_play,auto_speed
    d=request.json or{}
    if "enabled" in d: auto_play=d["enabled"]
    if "speed" in d: auto_speed=max(1,int(d["speed"]))
    return jsonify({"auto_play":auto_play,"speed":auto_speed})

@app.route("/api/game/reset",methods=["POST"])
def api_reset():
    global game,game_log,auto_play,game_metrics
    auto_play=False;game=GoGame(size=CFG["board_size"],komi=CFG["komi"],move_cap=CFG["move_cap"])
    game_log.clear();game_metrics=_new_game_metrics()
    return jsonify({"status":"reset"})

@app.route("/api/game/full_reset",methods=["POST"])
def api_full_reset():
    global game,game_log,auto_play,scoreboard,games_played,game_metrics,round_state,round_history,eval_scoreboard
    auto_play=False;game=GoGame(size=CFG["board_size"],komi=CFG["komi"],move_cap=CFG["move_cap"])
    game_log.clear();game_metrics=_new_game_metrics()
    g=players[WHITE]
    g.nodes.clear();g.edges.clear();g.archive.clear()
    from dkg_engine import MetabolicState,SelfModel,HeartbeatState
    g.metabolism=MetabolicState();g.self_model=SelfModel();g.heartbeat=HeartbeatState();g.tick_count=0
    scoreboard={"black_wins":0,"white_wins":0,"draws":0,"games":[],"total_games":0}
    eval_scoreboard={"games":[]}
    games_played=0;_save_sb(scoreboard);_save_eval_sb(eval_scoreboard)
    round_state={"round":1,"phase":"train","phase_num":1,"phase_label":"TRAIN","current_round_results":None}
    round_history=[];_save_round_history(round_history)
    initialize()
    return jsonify({"status":"full_reset"})

@app.route("/api/game/log")
def api_log(): return jsonify(game_log[-int(request.args.get("n",40)):])

@app.route("/api/game/players")
def api_players():
    g=players[WHITE]
    cog=g.get_cognitive_nodes(n=15)
    brain_data={"nodes":len(g.nodes),"edges":len(g.edges),"archived":len(g.archive),
        "cognitive":[{"type":n.type,"content":n.content[:150],"relevance":round(n.relevance_score,3),"critic_score":round(getattr(n,'critic_score',0.5),3)} for n in cog],
        "identity":g.identity_context(),"drives":g.drive_state(),
        "tools":[{"name":t.keywords[0] if t.keywords else t.id,"executions":t.total_executions,
                  "success":round(t.success_rate,2),"replay_value":round(t.replay_value,2)} for t in g.get_tools()],
        "conversations":len(g.nodes_of_type("conversation"))}
    return jsonify({"black":{"type":CFG.get("opponent_type","bare_llm")},"white":brain_data,"single_brain":True,"brain_plays":"white"})

@app.route("/api/game/graph/<color>")
def api_graph(color):
    g=players[WHITE]  # always the brain
    return jsonify(g.viz_data(full=request.args.get("full","false")=="true"))

# ═════════════════════════════════════════════════════════════════════════════
#  BIOPSY
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/api/game/biopsy")
def api_biopsy():
    L=[]
    L.append("╔══════════════════════════════════════════════════════════════╗")
    L.append("║  DKG GO v13 — Brain(W) + Heat + Critic + Plans vs Opp(B)    ║")
    L.append("╚══════════════════════════════════════════════════════════════╝")
    L.append(f"  Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"  Round     : {round_state['round']}  Phase: {round_state['phase_label']} ({round_state['phase_num']}/2)")
    L.append(f"  Game #    : {games_played}  Move: {game.move_count}  Turn: {'Black(H)' if game.current==BLACK else 'White(Brain)'}")
    b,w=game.score()
    L.append(f"  Score     : B={b:.1f}  W={w:.1f}  margin:{w-b:+.1f}")
    L.append(f"  Captures  : B:{game.captures[BLACK]}  W:{game.captures[WHITE]}")
    L.append(f"  Auto      : {auto_play}  Speed: {auto_speed}s")
    L.append("")

    met=game_metrics
    L.append("┌─── CURRENT GAME METRICS ────────────────────────────────────┐")
    n_moves=len(met["moves_log"])
    L.append(f"  Moves played   : {n_moves}")
    L.append(f"  Timeouts       : {met['timeouts']}")
    if met["timeout_details"]:
        for td in met["timeout_details"][-5:]:
            L.append(f"    ⏱ Move {td['move']} — {td['elapsed']}s — {td.get('llm_calls',0)}calls — ~{td.get('prompt_chars',0)//4}tok")
    L.append(f"  Parse failures : {met['parse_failures']}")
    L.append(f"  Retries        : {met['retries']}")
    L.append(f"  Fallbacks      : {met['fallbacks']}")
    if met["latencies"]:
        lats=met["latencies"]
        L.append(f"  Move latency   : avg={sum(lats)/len(lats):.1f}s  min={min(lats):.1f}s  max={max(lats):.1f}s")
    if met["parse_methods"]:
        L.append(f"  Parse methods  : {json.dumps(met['parse_methods'])}")

    pips=met.get("pipeline_steps",[])
    if pips:
        L.append(f"  ── Pipeline Breakdown ──")
        tool_times=[p.get("tool_select_time",0) for p in pips if p.get("tool_select_time",0)>0]
        know_times=[p.get("knowledge_select_time",0) for p in pips if p.get("knowledge_select_time",0)>0]
        move_times=[p.get("move_time",0) for p in pips if p.get("move_time",0)>0]
        total_calls=sum(p.get("llm_calls",0) for p in pips)
        L.append(f"  Total LLM calls: {total_calls}  ({total_calls/max(1,len(pips)):.1f}/move)")
        if tool_times: L.append(f"  Tool select    : avg={sum(tool_times)/len(tool_times):.1f}s  max={max(tool_times):.1f}s")
        if know_times: L.append(f"  Knowledge sel  : avg={sum(know_times)/len(know_times):.1f}s  max={max(know_times):.1f}s")
        if move_times: L.append(f"  Move generation: avg={sum(move_times)/len(move_times):.1f}s  max={max(move_times):.1f}s")
        all_tools=[t for p in pips for t in (p.get("tools_used") or [])]
        if all_tools:
            from collections import Counter
            tc=Counter(all_tools)
            L.append(f"  Tools used     : {dict(tc)}")
        all_lenses=[p.get("lens") for p in pips if p.get("lens")]
        if all_lenses:
            from collections import Counter
            lc=Counter(all_lenses)
            L.append(f"  Lenses used    : {dict(lc)}")
        td=met.get("tool_deltas",{})
        if td:
            L.append(f"  ── Tool Effectiveness ──")
            for tname,deltas in sorted(td.items()):
                avg_d=sum(deltas)/len(deltas) if deltas else 0
                L.append(f"  {tname:>20}: {len(deltas)}x  avg Δ={avg_d:+.1f}  total Δ={sum(deltas):+.1f}")
    L.append("└────────────────────────────────────────────────────────────┘")
    L.append("")

    L.append(game.board_string())
    L.append("")

    opp_n=CFG.get('opponent_type','bare_llm')
    L.append(f"┌─── TRAINING RECORD (Brain=White vs {opp_n}=Black) ────────┐")
    opp_label=CFG.get("opponent_type","bare_llm")
    L.append(f"  Brain(W) wins: {scoreboard['white_wins']}  {opp_label}(B) wins: {scoreboard['black_wins']}  Draws: {scoreboard['draws']}  ({scoreboard['total_games']} games)")
    for gr in scoreboard["games"][-8:]:
        margin=gr.get("margin",gr.get("white_score",0)-gr.get("black_score",0))
        to_str=f" {gr.get('timeouts',0)}to" if gr.get('timeouts',0)>0 else ""
        fb_str=f" {gr.get('fallbacks',0)}fb" if gr.get('fallbacks',0)>0 else ""
        L.append(f"    G{gr['game']}: {gr['winner']:>5}  B:{gr['black_score']:5.1f}  W:{gr['white_score']:5.1f}  m:{margin:+5.1f}  {gr['moves']}mv{to_str}{fb_str}")
    L.append("└────────────────────────────────────────────────────────────┘")
    L.append("")

    probe_games=eval_scoreboard.get("games",[])
    if probe_games:
        L.append(f"┌─── PROBE RESULTS (frozen brain White vs {opp_n} Black) ──┐")
        white_wins=sum(1 for e in probe_games if e.get("winner")=="white")
        margins=[e.get("margin",0) for e in probe_games]
        avg_m=round(sum(margins)/max(1,len(margins)),1)
        L.append(f"  White wins: {white_wins}/{len(probe_games)} ({round(white_wins/max(1,len(probe_games))*100)}%)  Avg margin: {avg_m:+.1f}")
        for e in probe_games[-10:]:
            is_win=e.get("winner")=="white"
            m=e.get("margin",0)
            L.append(f"    {'✓' if is_win else '✗'} R{e.get('round','?')}: {e['winner']:>5}  B:{e['black_score']:5.1f}  W:{e['white_score']:5.1f}  m:{m:+5.1f}  {e['moves']}mv")
        L.append("└────────────────────────────────────────────────────────────┘")
        L.append("")

    if round_history:
        L.append("┌─── ROUND HISTORY ───────────────────────────────────────────┐")
        for rh in round_history[-8:]:
            tr=rh.get("train") or {};pr=rh.get("probe") or {}
            pw="✓" if rh.get("white_won_probe") else "✗"
            pm=rh.get("probe_margin",0) or 0
            L.append(f"    R{rh['round']}: Train={tr.get('winner','?'):>5}  Probe={pr.get('winner','?'):>5}({pw}) m:{pm:+.1f}")
        last=round_history[-1]
        L.append(f"  Cumul: {last.get('cumulative_white_wins',0)}/{last.get('cumulative_probes',0)} wins  Avg margin: {last.get('avg_margin',0):+.1f}")
        L.append("└────────────────────────────────────────────────────────────┘")
        L.append("")

    g=players[WHITE]
    L.append("┌─── ◉ BRAIN (White) vs Opponent (Black) ─────────────────────┐")
    L.append(f"  Nodes: {len(g.nodes)}  Edges: {len(g.edges)}  Archived: {len(g.archive)}")
    L.append(f"  Drives: {g.drive_state()}")

    # ── Working strategy status ──
    active_ws=active_working_strategy(g)
    if active_ws:
        declared=active_ws.meta.get("declared_at_move",0)
        horizon=active_ws.meta.get("horizon",6)
        moves_elapsed=game.move_count-declared
        ds=active_ws.meta.get("declared_score",{})
        now_margin=round(game.score()[1]-game.score()[0],1)
        delta=round(now_margin-ds.get("margin",0),1)
        L.append(f"  📋 Active plan (h={horizon}, elapsed={moves_elapsed}): {active_ws.content[:100]}")
        L.append(f"       Margin when declared: {ds.get('margin',0):+.1f}  Now: {now_margin:+.1f}  Δ: {delta:+.1f}")

    ws_all=g.nodes_of_type(WORKING_STRATEGY)
    resolved=[n for n in ws_all if n.meta.get("state") in("resolved","distilled")]
    if resolved:
        resolved.sort(key=lambda n: n.meta.get("resolved_at_move",0),reverse=True)
        L.append(f"  📋 Recent resolved plans ({len(resolved)}):")
        for ws in resolved[:5]:
            outcome=ws.meta.get("outcome","?")
            d=ws.meta.get("margin_delta",0)
            moves_n=len(ws.meta.get("moves_under_plan",[]))
            mark={"succeeded":"✓","failed":"✗","neutral":"="}.get(outcome,"?")
            distilled=" [distilled]" if ws.meta.get("distilled") else ""
            L.append(f"      {mark} {outcome:>9} Δ={d:+5.1f} ({moves_n}mv){distilled}: {ws.content[:70]}")

    tools=g.get_tools()
    if tools:
        L.append(f"  Tools ({len(tools)}):")
        for t in tools:
            tname=t.keywords[0] if t.keywords else t.id
            L.append(f"    {tname}: {t.total_executions}x sr={t.success_rate:.0%} rv={t.replay_value:.2f} rel={t.relevance_score:.2f}")
    cog=g.get_cognitive_nodes(n=12)
    L.append(f"  Cognitive ({len(cog)}):")
    for c in cog:
        cs=getattr(c,'critic_score',0.5)
        ch=getattr(c,'critic_history',[])
        if ch:
            ch_str=f" cs={cs:.2f} ({sum(ch)}/{len(ch)})"
        else:
            ch_str=f" cs=0.50 (untested)"
        L.append(f"    [{c.type:>8}] r={c.relevance_score:.2f}{ch_str} | {c.content[:90]}")
    L.append("└────────────────────────────────────────────────────────────┘")
    L.append("")

    # ── CRITIC ACTIVITY ──
    cs=CRITIC_STATS
    L.append("┌─── 🧐 CRITIC ACTIVITY ──────────────────────────────────────┐")
    lc=cs["lifetime_calls"];gc=cs["game_calls"]
    if lc==0:
        L.append("  NO CRITIC CALLS YET — mechanism has not fired")
        L.append(f"  (enable_critic={CFG.get('enable_critic',True)}, timeout={CFG.get('timeout_critic',180)}s)")
    else:
        lm,lmi,lv,le,lto=cs["lifetime_matches"],cs["lifetime_misses"],cs["lifetime_vague"],cs["lifetime_errors"],cs["lifetime_timeouts"]
        gm,gmi,gv,ge,gto=cs["game_matches"],cs["game_misses"],cs["game_vague"],cs["game_errors"],cs["game_timeouts"]
        verd_total=lm+lmi+lv
        L.append(f"  Lifetime: {lc} calls  ({verd_total} verdicts, {le} errors, {lto} timeouts)")
        if verd_total>0:
            L.append(f"    Distribution: match={lm} ({lm*100//verd_total}%) miss={lmi} ({lmi*100//verd_total}%) vague={lv} ({lv*100//verd_total}%)")
        L.append(f"  This game: {gc} calls  ({gm} match, {gmi} miss, {gv} vague, {ge} err, {gto} timeout)")
        lats=cs["latencies"]
        if lats:
            L.append(f"  Latency: avg={sum(lats)/len(lats):.1f}s  max={max(lats):.1f}s  min={min(lats):.1f}s")
        by_src=cs["by_source"]
        src_str=", ".join(f"{k}={v}" for k,v in sorted(by_src.items()) if v>0)
        if src_str: L.append(f"  By source: {src_str}")
        rv=cs["recent_verdicts"]
        if rv:
            L.append(f"  Recent verdicts ({min(len(rv),8)}):")
            icon={"match":"✓","miss":"✗","vague":"?","error":"!"}
            for v in rv[-8:]:
                ic=icon.get(v["outcome"],"·")
                L.append(f"    {ic} {v['outcome']:5} ({v['chosen']:>6}) [{v['source']:>16}] {v['elapsed']:>5.1f}s  {v['claim'][:55]}")
    L.append("└────────────────────────────────────────────────────────────┘")
    L.append("")

    # ── HEAT ANALYSIS (latest) ──
    heat_reads=g.nodes_of_type(HEAT_READ)
    if heat_reads:
        heat_reads.sort(key=lambda n: n.meta.get("move_num",0),reverse=True)
        latest=heat_reads[0]
        L.append("┌─── 🔥 HEAT ANALYSIS (latest read) ──────────────────────────┐")
        mn=latest.meta.get("move_num","?")
        L.append(f"  Stored at move: {mn}   LLM analysis: {'yes' if latest.meta.get('llm_used') else 'fallback'}   ({latest.meta.get('elapsed',0)}s)")
        anch=latest.meta.get("anchors",[])
        if anch:
            L.append(f"  Top anchors ({len(anch)}):")
            for a in anch[:5]:
                pt=a.get("pt",[0,0]);h=a.get("h",0);comps=a.get("comps",{})
                if comps:
                    dom=max(comps.items(),key=lambda x:x[1])
                    L.append(f"    ({pt[0]},{pt[1]})  h={h:4.1f}  dom={dom[0]}  comps={comps}")
                else:
                    L.append(f"    ({pt[0]},{pt[1]})  h={h:4.1f}")
        L.append(f"  Analysis text:")
        for line in (latest.content[:400]).split("\n"):
            if line.strip(): L.append(f"    {line.strip()}")
        # Total heat reads in KG
        L.append(f"  Total heat reads in KG: {len(heat_reads)} active + {sum(1 for nid,nd in g.archive.items() if nd.get('type')==HEAT_READ)} archived")
        L.append("└────────────────────────────────────────────────────────────┘")
        L.append("")

    # ── KG FORMATION & TRAVERSAL ──
    L.append("┌─── 📊 KG FORMATION & TRAVERSAL ─────────────────────────────┐")
    by_type={}
    for n in g.nodes.values():
        by_type[n.type]=by_type.get(n.type,0)+1
    ordered=["identity","drive","goal","tool","conversation","insight","theory","strategy","working_strategy","meta"]
    type_line="  Nodes by type: "+", ".join(f"{t}:{by_type[t]}" for t in ordered if t in by_type)
    extras=[t for t in by_type if t not in ordered]
    if extras: type_line+=", "+", ".join(f"{t}:{by_type[t]}" for t in extras)
    L.append(type_line)
    L.append(f"  Edges: {len(g.edges)}  Archived: {len(g.archive)}  Tick: {g.tick_count}")

    # Working strategy stats
    ws_all=g.nodes_of_type(WORKING_STRATEGY)
    if ws_all:
        states={}
        outcomes={}
        for n in ws_all:
            s=n.meta.get("state","active")
            states[s]=states.get(s,0)+1
            o=n.meta.get("outcome")
            if o: outcomes[o]=outcomes.get(o,0)+1
        state_str=", ".join(f"{k}:{v}" for k,v in sorted(states.items()))
        L.append(f"  Working strategies: {len(ws_all)} total — {state_str}")
        if outcomes:
            out_str=", ".join(f"{k}:{v}" for k,v in sorted(outcomes.items()))
            distilled=sum(1 for n in ws_all if n.meta.get("distilled"))
            L.append(f"    Outcomes: {out_str}  |  distilled: {distilled}")
    else:
        L.append("  Working strategies: 0")

    # Cognitive-node critic health summary
    cog_all=[n for n in g.nodes.values() if n.type in(INSIGHT,THEORY,STRATEGY)]
    if cog_all:
        untested=sum(1 for n in cog_all if not getattr(n,'critic_history',[]))
        tested=len(cog_all)-untested
        high_rel_untested=sum(1 for n in cog_all if not getattr(n,'critic_history',[]) and n.relevance_score>=0.7)
        hammered=sum(1 for n in cog_all if getattr(n,'critic_score',0.5)<0.35 and len(getattr(n,'critic_history',[]))>=2)
        validated=sum(1 for n in cog_all if getattr(n,'critic_score',0.5)>0.7 and len(getattr(n,'critic_history',[]))>=2)
        L.append(f"  Cognitive nodes: {len(cog_all)} total  tested:{tested}  untested:{untested}  (high-rel-untested:{high_rel_untested})")
        L.append(f"    Critic health: hammered:{hammered}  validated:{validated}")

    # Lens distribution (this game)
    pips=met.get("pipeline_steps",[])
    if pips:
        from collections import Counter
        lens_c=Counter(p.get("lens") for p in pips if p.get("lens"))
        if lens_c:
            L.append(f"  Lenses this game: "+", ".join(f"{k}:{v}" for k,v in lens_c.most_common()))
        # Tool invocations and deltas
        tools_used=[t for p in pips for t in (p.get("tools_used") or [])]
        if tools_used:
            tc=Counter(tools_used)
            L.append(f"  Tool calls this game: "+", ".join(f"{k}:{v}" for k,v in tc.most_common()))
        # Graph expansion counts
        expanded=sum(p.get("graph_expanded",0) for p in pips)
        if expanded>0:
            all_conns=[c for p in pips for c in (p.get("graph_connections") or [])]
            L.append(f"  Graph expansions this game: {expanded} nodes added via edges  ({len(all_conns)} traversals)")
    L.append("└────────────────────────────────────────────────────────────┘")
    L.append("")

    snaps=_list_snaps()
    if snaps: L.append(f"Snapshots ({len(snaps)}): {', '.join(s['label'] for s in snaps[-12:])}")
    L.append("═══════════════════════════ END BIOPSY ═══════════════════════")
    return jsonify({"biopsy":"\n".join(L)})

@app.route("/api/round")
def api_round():
    return jsonify({"round":round_state["round"],"phase":round_state["phase"],
        "phase_num":round_state["phase_num"],"phase_label":round_state["phase_label"],
        "current_results":round_state.get("current_round_results"),"history":round_history[-20:]})

@app.route("/api/experiment/snapshots")
def api_snapshots(): return jsonify(_list_snaps())

@app.route("/api/experiment/snapshot",methods=["POST"])
def api_take_snapshot():
    label=(request.json or{}).get("label",f"g{games_played}")
    _save_snap("brain",label)
    return jsonify({"status":"saved","label":label,"game":games_played})

@app.route("/api/experiment/eval_scoreboard")
def api_eval_scoreboard(): return jsonify(eval_scoreboard)

@app.route("/api/experiment/round_history")
def api_round_history(): return jsonify(round_history)

@app.route("/api/config")
def api_config(): return jsonify(CFG)

@app.route("/api/config",methods=["POST"])
def api_config_update():
    global CFG
    d=request.json or{}
    for k,v in d.items():
        if k in CFG and not k.startswith("//"):
            CFG[k]=v
    with open(CFG_PATH,"w") as f: json.dump(CFG,f,indent=2)
    return jsonify({"status":"updated","config":CFG})

# ═════════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    import sys
    port=CFG["port"]
    if len(sys.argv)>1: port=int(sys.argv[1])
    print("\n"+"="*60)
    print("  DKG Go v13 — Brain (White) vs Opponent (Black)")
    print("  Heat Map → Position Read → Tools → Lens → Knowledge → Plan → Move")
    print(f"  Opponent: {CFG.get('opponent_type','bare_llm')}  Board: {CFG['board_size']}x{CFG['board_size']}  Komi: {CFG['komi']}")
    print(f"  Critic: {'on' if CFG.get('enable_critic',True) else 'off'}  WS: {'on' if CFG.get('enable_working_strategy',True) else 'off'}  Heat: {'on' if CFG.get('enable_heat_analysis',True) else 'off'}")
    print("="*60+"\n")
    preflight_check()
    initialize()
    game_metrics["_start_time"]=time.time()
    bg_thread.start()
    print(f"\n  -> http://localhost:{port}\n")
    app.run(host="0.0.0.0",port=port,debug=False)
