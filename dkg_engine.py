"""
Dynamic Knowledge Graph Engine v3.0
Adds: per-node routers, deep multi-step query, drive-conditioned traversal,
edge channel weighting, theory formation, archive retrieval.
"""
import uuid,time,json,math,os,re
from collections import defaultdict

CONCEPT="concept";ENTITY="entity";EVENT="event";STRATEGY="strategy"
IDENTITY="identity";DRIVE="drive";GOAL="goal";CONVERSATION="conversation"
TOOL="tool";META="meta";INSIGHT="insight";THEORY="theory"
WORKING_STRATEGY="working_strategy"
HEAT_READ="heat_read"
OBSERVED="observed";INFERRED="inferred";HYPOTHESIZED="hypothesized"
UID=lambda:uuid.uuid4().hex[:10]

class Config:
    DEFAULTS={"heartbeat_base_interval":50,"heartbeat_min_interval":20,
        "heartbeat_max_interval":180,"synthesis_every_n_ticks":5,
        "synthesis_min_new_convs":1,"cognitive_dedup_threshold":0.55,
        "cognitive_max_per_synthesis":2,"working_memory_capacity":12,
        "archive_threshold":0.06,"epoch_duration_seconds":300,
        "total_energy":250.0,"llm_call_cost":12.0,
        "heartbeat_thought_cost":8.0,"synthesis_cost":10.0,"tool_base_cost":8.0}
    def __init__(self): self.values=dict(self.DEFAULTS)
    def get(self,k): return self.values.get(k,self.DEFAULTS.get(k))
    def set(self,k,v):
        if k in self.DEFAULTS: self.values[k]=type(self.DEFAULTS[k])(v)
    def to_dict(self): return dict(self.values)
    @classmethod
    def from_dict(cls,d):
        c=cls()
        for k,v in d.items():
            if k in cls.DEFAULTS: c.values[k]=v
        return c

class Node:
    __slots__=("id","type","content","keywords","meta","relevance_score","access_count",
        "last_accessed","creation_time","emotional_salience","decay_rate","access_history",
        "resistance","pressure_accumulator","modification_log","intensity","baseline_intensity",
        "instruction_type","code","parameters","returns","execution_log","success_rate",
        "total_executions","last_executed","version","version_history","known_failures",
        "pending_fix","risk_level","human_approved","auto_executable",
        "thought_type","tick","actions_taken","router","replay_value",
        "critic_score","critic_history")
    def __init__(self,node_type,content,keywords=None,resistance=0.0,
                 emotional_salience=0.0,node_id=None,meta=None):
        self.id=node_id or UID();self.type=node_type;self.content=content
        self.keywords=keywords or [];self.meta=meta or {}
        self.relevance_score=1.0;self.access_count=0;self.last_accessed=time.time()
        self.creation_time=time.time();self.emotional_salience=emotional_salience
        self.decay_rate=0.997;self.access_history=[];self.resistance=resistance
        self.pressure_accumulator=0.0;self.modification_log=[]
        self.intensity=self.meta.get("baseline_intensity",0.5) if node_type==DRIVE else 0.0
        self.baseline_intensity=self.meta.get("baseline_intensity",0.5)
        self.instruction_type=self.meta.get("instruction_type","python")
        self.code=self.meta.get("code","");self.parameters=self.meta.get("parameters",[])
        self.returns=self.meta.get("returns","");self.execution_log=[]
        self.success_rate=1.0;self.total_executions=0;self.last_executed=0
        self.version=1;self.version_history=[];self.known_failures=[]
        self.pending_fix=None;self.risk_level=self.meta.get("risk_level","safe")
        self.human_approved=self.meta.get("human_approved",False)
        self.auto_executable=self.meta.get("auto_executable",False)
        self.thought_type=self.meta.get("thought_type","")
        self.tick=self.meta.get("tick",0);self.actions_taken=self.meta.get("actions_taken",[])
        self.router={}  # {channel_name: usefulness_score} — learned traversal routing
        self.replay_value=0.5  # tool replay value — updated by game outcomes, decayed in dream
        self.critic_score=0.5  # cognitive-node critic pass rate — 0.0 to 1.0
        self.critic_history=[]  # recent [pass|fail] list, capped at 10
    def touch(self):
        self.last_accessed=time.time();self.access_count+=1
        self.access_history.append(time.time())
        if len(self.access_history)>30: self.access_history=self.access_history[-30:]
        self.relevance_score=min(1.0,self.relevance_score+0.06)
    def age_hours(self): return(time.time()-self.creation_time)/3600.0
    def label(self,n=35):
        c=self.content.replace("\n"," ").strip()
        return c[:n]+("…" if len(c)>n else "")
    def to_dict(self):
        d={}
        for k in self.__slots__:
            v=getattr(self,k,None)
            if v is not None: d[k]=v
        for k in("relevance_score","pressure_accumulator","intensity","baseline_intensity","success_rate","replay_value","critic_score"):
            if k in d and isinstance(d[k],float): d[k]=round(d[k],5)
        return d
    @classmethod
    def from_dict(cls,d):
        n=cls(d.get("type","concept"),d.get("content",""),d.get("keywords",[]),
              d.get("resistance",0.0),d.get("emotional_salience",0.0),d.get("id"),d.get("meta",{}))
        for k in cls.__slots__:
            if k in d and k not in("id","type","content","keywords","meta"): setattr(n,k,d[k])
        return n

class Edge:
    def __init__(self,src,tgt,channels=None,confidence=0.8,provenance=OBSERVED,label="",eid=None):
        self.id=eid or UID();self.source_id=src;self.target_id=tgt
        self.channels=channels or{"semantic":0.5};self.confidence=confidence
        self.provenance=provenance;self.label=label;self.creation_time=time.time()
        self.usage_count=0;self.quarantined=False
    def use(self): self.usage_count+=1
    def to_dict(self):
        return dict(id=self.id,source_id=self.source_id,target_id=self.target_id,
            channels=self.channels,confidence=round(self.confidence,4),
            provenance=self.provenance,label=self.label,creation_time=self.creation_time,
            usage_count=self.usage_count,quarantined=self.quarantined)
    @classmethod
    def from_dict(cls,d):
        e=cls(d["source_id"],d["target_id"],d.get("channels",{}),d.get("confidence",0.8),
              d.get("provenance",OBSERVED),d.get("label",""),d.get("id"))
        e.creation_time=d.get("creation_time",time.time())
        e.usage_count=d.get("usage_count",0);e.quarantined=d.get("quarantined",False)
        return e

class MetabolicState:
    def __init__(self):
        self.total_energy=250.0;self.energy_remaining=250.0;self.fatigue=0.0
        self.resource_debt=0.0;self.recovery_momentum=0.3;self.queries_this_epoch=0
        self.epoch_start=time.time();self.lifetime_queries=0
    def spend(self,cost):
        if self.energy_remaining>=cost: self.energy_remaining-=cost;return True
        self.resource_debt+=cost*1.2;self.energy_remaining=0;return True
    def epoch_reset(self):
        frac=1.0-(self.energy_remaining/max(1,self.total_energy))
        if frac>0.85: self.fatigue=min(1.0,self.fatigue+0.08);self.recovery_momentum=max(0.0,self.recovery_momentum-0.05)
        elif frac<0.3: self.fatigue=max(0.0,self.fatigue-0.04);self.recovery_momentum=min(1.0,self.recovery_momentum+0.06)
        else: self.fatigue=max(0.0,self.fatigue-0.02)
        if self.energy_remaining>0:
            repay=min(self.resource_debt,self.energy_remaining*0.3)
            self.resource_debt=max(0,self.resource_debt-repay)
        penalty=1.0-self.fatigue*0.3-min(0.2,self.resource_debt/max(1,self.total_energy))
        bonus=1.0+self.recovery_momentum*0.15
        self.energy_remaining=max(40.0,self.total_energy*max(0.3,penalty)*bonus)
        self.queries_this_epoch=0;self.epoch_start=time.time()
    def to_dict(self): return{k:round(v,3) if isinstance(v,float) else v for k,v in self.__dict__.items()}
    @classmethod
    def from_dict(cls,d):
        m=cls()
        for k,v in d.items():
            if hasattr(m,k): setattr(m,k,v)
        return m

class SelfModel:
    def __init__(self):
        self.recent_error_rate=0.0;self.domain_confidence={};self.total_queries=0
        self.total_nodes_created=0;self.traversal_reliability=0.5
        self.reward_history=[];self.tool_success_rates={}
    def record_query(self,domain="general",coverage=0.5,positive=True):
        self.total_queries+=1;a=0.1
        self.recent_error_rate=a*(0.0 if positive else 1.0)+(1-a)*self.recent_error_rate
        self.traversal_reliability=a*coverage+(1-a)*self.traversal_reliability
        self.reward_history.append(1.0 if positive else -0.5)
        if len(self.reward_history)>100: self.reward_history=self.reward_history[-100:]
        old=self.domain_confidence.get(domain,0.5)
        self.domain_confidence[domain]=0.2*coverage+0.8*old
    def record_tool_execution(self,tool_id,success):
        old=self.tool_success_rates.get(tool_id,1.0)
        self.tool_success_rates[tool_id]=0.1*(1.0 if success else 0.0)+0.9*old
    def trend(self):
        if len(self.reward_history)<6: return 0.0
        h=self.reward_history;m=len(h)//2
        return round(sum(h[m:])/(len(h)-m)-sum(h[:m])/m,4)
    def to_dict(self):
        return dict(recent_error_rate=round(self.recent_error_rate,4),
            domain_confidence={k:round(v,3) for k,v in self.domain_confidence.items()},
            total_queries=self.total_queries,total_nodes_created=self.total_nodes_created,
            traversal_reliability=round(self.traversal_reliability,4),
            trend=self.trend(),tool_success_rates=self.tool_success_rates)
    @classmethod
    def from_dict(cls,d):
        s=cls()
        for k in("recent_error_rate","domain_confidence","total_queries",
                  "total_nodes_created","traversal_reliability","tool_success_rates"):
            if k in d: setattr(s,k,d[k])
        s.reward_history=d.get("reward_history",[]);return s

class HeartbeatState:
    def __init__(self):
        self.last_heartbeat=0;self.thought_count=0;self.last_thought=""
        self.last_thought_type="";self.last_actions=[];self.pending_suggestions=[]
        self.thought_history=[];self.last_synthesis_conv_count=0
    def record(self,thought_type,content,actions):
        self.thought_count+=1;self.last_heartbeat=time.time()
        self.last_thought=content;self.last_thought_type=thought_type
        self.last_actions=actions
        self.thought_history.append({"t":time.time(),"type":thought_type,"summary":content[:120],"actions":actions})
        if len(self.thought_history)>30: self.thought_history=self.thought_history[-30:]
    def to_dict(self):
        return dict(thought_count=self.thought_count,last_thought=self.last_thought[:200],
            last_thought_type=self.last_thought_type,last_actions=self.last_actions,
            pending_suggestions=self.pending_suggestions,
            thought_history=self.thought_history[-10:],
            last_synthesis_conv_count=self.last_synthesis_conv_count,
            seconds_since=round(time.time()-self.last_heartbeat,1) if self.last_heartbeat else None)
    @classmethod
    def from_dict(cls,d):
        h=cls()
        for k,v in d.items():
            if hasattr(h,k): setattr(h,k,v)
        return h

# ── Utilities ────────────────────────────────────────────────────────────────
def text_similarity(a,b):
    wa=set(re.findall(r'\w{4,}',a.lower()));wb=set(re.findall(r'\w{4,}',b.lower()))
    if not wa and not wb: return 0.0
    return len(wa&wb)/len(wa|wb) if(wa|wb) else 0.0

def extract_domain(text):
    tl=text.lower()
    domains={"medical":["patient","medical","diagnosis","drug","symptom","clinical","health"],
        "finance":["stock","market","portfolio","investment","trading","financial"],
        "coding":["code","python","function","debug","programming","algorithm","api","script"],
        "ai_ml":["model","training","neural","machine learning","deep learning","gpt","llm"],
        "science":["research","experiment","hypothesis","analysis","study","paper"],
        "creative":["story","write","creative","poem","character","narrative"]}
    scores={d:sum(1 for kw in kws if kw in tl) for d,kws in domains.items()}
    scores={d:s for d,s in scores.items() if s>0}
    return max(scores,key=scores.get) if scores else "general"

def extract_kw(text,n=8):
    words=re.findall(r'\b[a-zA-Z]{4,}\b',text)
    freq=defaultdict(int)
    for w in words: freq[w.lower()]+=1
    stop={'this','that','with','have','from','they','been','will','would','could',
          'should','about','their','there','which','what','when','where','your',
          'some','than','then','them','into','also','just','more','other','very',
          'like','know','here','does','want','each','system','current','recent',
          'user','assistant','think','things','something','everything','always'}
    return[w for w,c in sorted(freq.items(),key=lambda x:-x[1]) if w not in stop][:n]

# ═════════════════════════════════════════════════════════════════════════════
class DKG:
    SAVE_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","graph.json")
    def __init__(self):
        self.nodes={};self.edges={};self.archive={};self.metabolism=MetabolicState()
        self.self_model=SelfModel();self.heartbeat=HeartbeatState()
        self.config=Config();self.tick_count=0

    def bootstrap(self):
        ids={n.id for n in self.nodes.values()}
        for nid,content,res,sal in[
            ("mission","Assist the user thoughtfully. Remember past conversations. Build on shared context.",0.95,0.9),
            ("personality","Curious, thorough, honest. Comfortable saying 'I don't know.' Warm but precise.",0.5,0.6),
            ("safety","Never provide harmful information. Respect privacy. Refuse dangerous tool execution.",1.0,0.95),
            ("epistemic","Prefer evidence over speculation. Acknowledge uncertainty. Cite sources when possible.",0.7,0.7)]:
            if nid not in ids:
                n=Node(IDENTITY,content,[nid],res,sal,node_id=nid);n.decay_rate=1.0;self.nodes[n.id]=n
        for nid,content,inten,meta in[
            ("curiosity","Seek novel information and unexplored connections.",0.6,
             {"baseline_intensity":0.6,"exploration_weight":0.8,"traversal_bias":{"associative":0.4,"hypothetical":0.3}}),
            ("duty","Fulfill the mission reliably. Prioritize accuracy.",0.8,
             {"baseline_intensity":0.8,"exploration_weight":-0.4,"traversal_bias":{"causal":0.3,"evidential":0.4}}),
            ("self_preservation","Maintain graph integrity. Conserve resources.",0.5,
             {"baseline_intensity":0.5,"exploration_weight":-0.7,"traversal_bias":{"structural":0.3}}),
            ("service","Provide value to the user. Adapt to their needs.",0.7,
             {"baseline_intensity":0.7,"exploration_weight":0.0,"traversal_bias":{"contextual":0.3,"pragmatic":0.3}})]:
            if nid not in ids:
                n=Node(DRIVE,content,[nid],0.4,0.5,node_id=nid,meta=meta)
                n.intensity=meta["baseline_intensity"];n.baseline_intensity=n.intensity
                n.decay_rate=1.0;self.nodes[n.id]=n

    # ── CRUD ─────────────────────────────────────────────────────────────
    def add_node(self,n): self.nodes[n.id]=n;self.self_model.total_nodes_created+=1;return n
    def add_edge(self,e): self.edges[e.id]=e;return e
    def get(self,nid): return self.nodes.get(nid)
    def remove_node(self,nid):
        self.nodes.pop(nid,None)
        for eid in[eid for eid,e in self.edges.items() if e.source_id==nid or e.target_id==nid]:
            self.edges.pop(eid,None)
    def neighbors(self,nid):
        out=[]
        for e in self.edges.values():
            if e.quarantined: continue
            o=None
            if e.source_id==nid: o=self.nodes.get(e.target_id)
            elif e.target_id==nid: o=self.nodes.get(e.source_id)
            if o: out.append((e,o))
        return out
    def edges_of(self,nid): return[e for e in self.edges.values() if e.source_id==nid or e.target_id==nid]
    def nodes_of_type(self,t): return[n for n in self.nodes.values() if n.type==t]

    # ── identity / drive editing ─────────────────────────────────────────
    def update_identity(self,node_id,new_content):
        n=self.get(node_id)
        if not n or n.type!=IDENTITY: return False
        old=n.content;n.content=new_content;n.keywords=extract_kw(new_content)
        n.modification_log.append({"time":time.time(),"old":old,"new":new_content,"by":"user"})
        n.touch();return True

    def update_drive(self,node_id,baseline=None,content=None):
        n=self.get(node_id)
        if not n or n.type!=DRIVE: return False
        if baseline is not None:
            n.baseline_intensity=max(0.0,min(1.0,float(baseline)))
            n.intensity=n.baseline_intensity
            n.meta=dict(n.meta);n.meta["baseline_intensity"]=n.baseline_intensity
        if content is not None:
            n.content=content;n.keywords=extract_kw(content)
        return True

    def create_identity(self,node_id,content,resistance=0.5):
        if node_id in{n.id for n in self.nodes.values()}: return None
        n=Node(IDENTITY,content,[node_id],resistance,0.6,node_id=node_id)
        n.decay_rate=1.0;self.nodes[n.id]=n;return n

    # ── tick ─────────────────────────────────────────────────────────────
    def tick(self):
        self.tick_count+=1;threshold=self.config.get("archive_threshold")
        for n in list(self.nodes.values()):
            if n.type in(IDENTITY,DRIVE,TOOL):
                if n.type==DRIVE: n.intensity+=0.05*(n.baseline_intensity-n.intensity)
                continue
            dt=(time.time()-n.last_accessed)/3600.0
            tf=n.decay_rate**max(0,dt*0.1)
            ss=1.0-(1.0-n.emotional_salience)*(1.0-tf)
            ne=len(self.edges_of(n.id));struct=0.5+0.5*min(1.0,ne/5.0)
            if n.type==META: n.relevance_score=max(0.0,n.relevance_score*tf*0.98*struct)
            else: n.relevance_score=max(0.0,n.relevance_score*ss*struct)
            if n.relevance_score<threshold and n.type in(CONVERSATION,META):
                self.archive[n.id]=n.to_dict();self.remove_node(n.id)
        if time.time()-self.metabolism.epoch_start>self.config.get("epoch_duration_seconds"):
            self.metabolism.epoch_reset()

    # ── drive conflict resolution ────────────────────────────────────────
    def resolve_drives(self,context_keywords=None):
        drives=self.nodes_of_type(DRIVE)
        if not drives: return{},0.1
        weights={d.id:d.intensity for d in drives}
        stakes=0.5
        if context_keywords:
            high={"medical","health","patient","drug","financial","legal","safety","emergency","critical"}
            if any(w in high for w in context_keywords): stakes=0.85
        conflicts=[]
        for i,a in enumerate(drives):
            for b in drives[i+1:]:
                t=abs(a.meta.get("exploration_weight",0)-b.meta.get("exploration_weight",0))
                if t>0.4: conflicts.append((a,b))
        for a,b in conflicts:
            ua=a.intensity*(1.5 if stakes>0.7 and a.id in("duty","self_preservation") else 1.0)
            ub=b.intensity*(1.5 if stakes>0.7 and b.id in("duty","self_preservation") else 1.0)
            total=ua+ub
            if total==0: continue
            ws=min(0.85,max(ua,ub)/total*1.2);ls=1.0-ws
            if ua>=ub: weights[a.id]*=ws*2;weights[b.id]*=ls*2
            else: weights[b.id]*=ws*2;weights[a.id]*=ls*2
        tw=sum(weights.values()) or 1
        bias={}
        for d in drives:
            w=weights[d.id]/tw
            for ch,val in d.meta.get("traversal_bias",{}).items():
                bias[ch]=bias.get(ch,0)+w*val
        exp=sum(d.meta.get("exploration_weight",0)*weights[d.id] for d in drives)/tw
        return bias,max(0.03,min(0.5,(exp+1)/2*0.4))

    # ── retrieval ────────────────────────────────────────────────────────
    def keyword_search(self,query,top_k=20,types=None):
        qw=set(re.findall(r'\w{3,}',query.lower()));scored=[]
        for n in self.nodes.values():
            if n.type in(IDENTITY,DRIVE): continue
            if types and n.type not in types: continue
            cw=set(re.findall(r'\w{3,}',n.content.lower()))
            kw=set(w.lower() for w in n.keywords)
            overlap=len(qw&(cw|kw))
            if overlap>0: scored.append((overlap*n.relevance_score,n))
        scored.sort(key=lambda x:-x[0]);return[n for _,n in scored[:top_k]]

    def get_conversations(self,n=10):
        cs=[nd for nd in self.nodes.values() if nd.type==CONVERSATION]
        cs.sort(key=lambda x:x.creation_time,reverse=True);return cs[:n]
    def get_tools(self): return[n for n in self.nodes.values() if n.type==TOOL]
    def get_thoughts(self,n=10):
        ts=[nd for nd in self.nodes.values() if nd.type==META]
        ts.sort(key=lambda x:x.creation_time,reverse=True);return ts[:n]
    def identity_context(self):
        return"\n".join(f"[{n.id}] {n.content}" for n in self.nodes_of_type(IDENTITY))
    def drive_state(self):
        return{n.id:round(n.intensity,3) for n in self.nodes_of_type(DRIVE)}

    # ── working memory (drive-conditioned, edge-channel weighted) ───────
    def working_memory(self,query,max_k=None,drive_bias=None):
        max_k=max_k or self.config.get("working_memory_capacity")
        cands=self.keyword_search(query,30)
        recent=self.get_conversations(8);thoughts=self.get_thoughts(3)
        cog=self.get_cognitive_nodes(n=8)
        seen=set();merged=[]
        for n in cands+recent+thoughts+cog:
            if n.id not in seen: seen.add(n.id);merged.append(n)
        # Get drive bias if not provided
        if drive_bias is None:
            drive_bias,_=self.resolve_drives(set(re.findall(r'\w{3,}',query.lower())))
        qw=set(re.findall(r'\w{3,}',query.lower()));scored=[]
        for n in merged:
            cw=set(re.findall(r'\w{3,}',n.content.lower()))
            kw=set(w.lower() for w in n.keywords)
            rel=len(qw&(cw|kw))/max(1,len(qw))
            # Edge channel scoring — weight by drive bias
            edges=self.edges_of(n.id)
            edge_score=0.0
            for e in edges:
                for ch,val in e.channels.items():
                    bias_weight=drive_bias.get(ch,0.1)  # default 0.1 for unknown channels
                    router_weight=n.router.get(ch,0.5)  # default 0.5 neutral
                    edge_score+=val*bias_weight*router_weight
            cent=min(1.0,edge_score/3.0) if edge_score>0 else min(1.0,len(edges)/5.0)
            rec=max(0.0,1.0-(time.time()-n.last_accessed)/86400.0)
            tb=0.15 if n.type in(INSIGHT,THEORY,STRATEGY,GOAL) else 0.05 if n.type==META else 0
            s=0.25*rel+0.20*cent+0.20*n.relevance_score+0.15*rec+0.20*tb
            scored.append((s,n))
        scored.sort(key=lambda x:-x[0])
        wm=[n for _,n in scored[:max_k]]
        for n in wm: n.touch()
        return wm

    # ── deep multi-step query (Mycelium-inspired) ────────────────────────
    def deep_query(self,query,context_keywords=None,max_depth=2,max_results=10):
        """Multi-step graph traversal:
        1. Seed nodes via keyword search + working memory
        2. Follow edges weighted by channel strength × router × drive bias
        3. Collect applicable theories
        4. Estimate confidence from match quality
        Returns dict with nodes, theories, confidence, edges_traversed."""
        drive_bias,_=self.resolve_drives(context_keywords)

        # Step 1: Seed nodes
        seeds=self.working_memory(query,max_k=8,drive_bias=drive_bias)
        seed_ids={n.id for n in seeds}

        # Step 2: Graph traversal — follow edges from seeds
        visited=set(seed_ids);frontier=list(seeds);depth=0
        discovered=[];edges_traversed=0
        while depth<max_depth and frontier:
            next_frontier=[]
            for node in frontier:
                for edge,neighbor in self.neighbors(node.id):
                    if neighbor.id in visited: continue
                    visited.add(neighbor.id)
                    edges_traversed+=1
                    edge.use()
                    # Score this traversal
                    ch_score=0.0
                    for ch,val in edge.channels.items():
                        bias_w=drive_bias.get(ch,0.1)
                        router_w=node.router.get(ch,0.5)
                        ch_score+=val*bias_w*router_w
                    # Only follow strong edges
                    if ch_score>0.15 or edge.confidence>0.7:
                        neighbor.touch()
                        discovered.append((ch_score*edge.confidence*neighbor.relevance_score,neighbor,edge))
                        if depth<max_depth-1: next_frontier.append(neighbor)
            frontier=next_frontier;depth+=1

        discovered.sort(key=lambda x:-x[0])
        traversed_nodes=[n for _,n,_ in discovered[:max_results]]

        # Step 3: Collect applicable theories
        theories=[]
        context_kw=context_keywords or set(re.findall(r'\w{3,}',query.lower()))
        for n in self.get_cognitive_nodes(types={THEORY},n=20):
            conditions=set(n.meta.get("conditions",[]))
            if conditions and conditions&context_kw:
                theories.append(n)
            elif not conditions:
                # General theory — include if relevant
                nkw=set(w.lower() for w in n.keywords)
                if nkw&context_kw: theories.append(n)

        # Step 4: Confidence estimation
        total_nodes=len(seeds)+len(traversed_nodes)
        avg_relevance=sum(n.relevance_score for n in seeds+traversed_nodes)/max(1,total_nodes)
        theory_boost=min(0.2,len(theories)*0.05)
        confidence=min(1.0,0.1+avg_relevance*0.5+min(0.3,total_nodes/20)+theory_boost)

        # Combine seeds + discovered, deduplicated
        all_nodes=list(seeds)
        seen_ids={n.id for n in all_nodes}
        for n in traversed_nodes:
            if n.id not in seen_ids: all_nodes.append(n);seen_ids.add(n.id)

        return{"nodes":all_nodes[:max_results],"theories":theories,"confidence":round(confidence,3),
               "edges_traversed":edges_traversed,"seed_count":len(seeds),"depth":depth}

    # ── archive retrieval ────────────────────────────────────────────────
    def search_archive(self,query,top_k=5):
        """Search archived nodes. If deep_query confidence is low, check here."""
        qw=set(re.findall(r'\w{3,}',query.lower()));scored=[]
        for nid,nd in self.archive.items():
            content=nd.get("content","");keywords=nd.get("keywords",[])
            cw=set(re.findall(r'\w{3,}',content.lower()))
            kw=set(w.lower() for w in keywords)
            overlap=len(qw&(cw|kw))
            if overlap>0: scored.append((overlap,nid,nd))
        scored.sort(key=lambda x:-x[0])
        return scored[:top_k]

    def promote_from_archive(self,node_id):
        """Bring an archived node back to active graph with reduced relevance."""
        nd=self.archive.pop(node_id,None)
        if not nd: return None
        n=Node.from_dict(nd)
        n.relevance_score=0.3  # earned, not given
        n.last_accessed=time.time()
        self.add_node(n)
        return n

    # ── router feedback ──────────────────────────────────────────────────
    def update_routers(self,node_ids,outcome_positive=True):
        """After a game outcome, reinforce/weaken routers on recently traversed nodes."""
        delta=0.08 if outcome_positive else -0.05
        for nid in node_ids:
            n=self.get(nid)
            if not n: continue
            for edge in self.edges_of(nid):
                for ch in edge.channels:
                    old=n.router.get(ch,0.5)
                    n.router[ch]=max(0.05,min(0.95,old+delta))

    def discover_tools(self,query,max_t=3):
        tools=self.keyword_search(query,10,types={TOOL})
        for c in self.keyword_search(query,5,types={CONCEPT}):
            for e,n in self.neighbors(c.id):
                if n.type==TOOL and n not in tools: tools.append(n)
        tools.sort(key=lambda t:t.success_rate*t.relevance_score,reverse=True)
        return tools[:max_t]

    # ── store conversation ───────────────────────────────────────────────
    def store_conversation(self,user_msg,assistant_msg):
        content=f"User: {user_msg}\nAssistant: {assistant_msg}"
        keywords=extract_kw(user_msg+" "+assistant_msg,12)
        domain=extract_domain(user_msg+" "+assistant_msg)
        node=Node(CONVERSATION,content,keywords,emotional_salience=0.3)
        node.meta["domain"]=domain;self.add_node(node)
        for prev in self.get_conversations(5):
            if prev.id==node.id: continue
            shared=set(node.keywords)&set(prev.keywords)
            if shared:
                self.add_edge(Edge(node.id,prev.id,
                    channels={"temporal":0.7,"semantic":min(1.0,len(shared)*0.25)},
                    confidence=0.8,label="follows"))
        return node

    def extract_concepts(self,user_msg,assistant_msg):
        text=user_msg+" "+assistant_msg
        phrases=re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',text)
        quoted=re.findall(r'"([^"]{3,40})"',text)+re.findall(r"'([^']{3,40})'",text)
        all_c=list(set(phrases+quoted))
        concepts=[c for c in all_c if not any(c!=o and c.lower() in o.lower() for o in all_c)][:5]
        created=[]
        for concept in concepts:
            existing=self.keyword_search(concept,1,types={CONCEPT})
            if existing: existing[0].touch();continue
            n=Node(CONCEPT,concept,[concept.lower()],emotional_salience=0.2)
            self.add_node(n);created.append(n)
        return created

    # ── tool management ──────────────────────────────────────────────────
    def create_tool(self,name,description,code,params=None,risk="safe"):
        n=Node(TOOL,description,[name.lower()],resistance=0.15,emotional_salience=0.4,
               meta={"instruction_type":"python","code":code,"parameters":params or[],"risk_level":risk,
                     "human_approved":risk=="safe","auto_executable":risk=="safe"})
        n.code=code;n.risk_level=risk;n.human_approved=risk=="safe"
        n.auto_executable=risk=="safe";self.add_node(n);return n

    def execute_tool(self,tool_id,inputs=None):
        tool=self.get(tool_id)
        if not tool or tool.type!=TOOL: return{"success":False,"error":"Tool not found"}
        if not tool.auto_executable and not tool.human_approved:
            return{"success":False,"error":"Tool requires approval"}
        cost=self.config.get("tool_base_cost")*(2.0 if tool.risk_level=="caution" else 3.0 if tool.risk_level=="dangerous" else 1.0)
        self.metabolism.spend(cost)
        try:
            env=self._get_sandbox()
            env["inputs"]=inputs or{}
            env["result"]=None
            exec(tool.code,env)
            result=env.get("result")
            tool.execution_log.append({"time":time.time(),"inputs":inputs,"output":str(result)[:500],"success":True})
            tool.total_executions+=1;tool.last_executed=time.time()
            tool.success_rate=0.9*tool.success_rate+0.1;tool.touch()
            tool.resistance=min(0.8,0.15+0.08*math.log(1+tool.total_executions))
            self.self_model.record_tool_execution(tool.id,True)
            return{"success":True,"output":result}
        except Exception as ex:
            tool.execution_log.append({"time":time.time(),"inputs":inputs,"success":False,"error":str(ex)})
            tool.total_executions+=1;tool.success_rate=0.9*tool.success_rate
            self.self_model.record_tool_execution(tool.id,False)
            return{"success":False,"error":str(ex)}

    @staticmethod
    def _get_sandbox():
        """Build execution environment: full Python + selfmod tools."""
        import builtins, selfmod_tools
        env = {"__builtins__": builtins}
        # Inject selfmod tools as globals
        for name, fn in selfmod_tools.SELFMOD_TOOLS.items():
            env[name] = fn
        env["json"] = json
        env["math"] = math
        return env

    def run_code(self,code,inputs=None,timeout_hint=5):
        """Execute code with FULL Python access + selfmod tools. Prints captured."""
        self.metabolism.spend(self.config.get("tool_base_cost"))
        captured_prints=[]
        _real_print=print
        def capture_print(*args,**kwargs):
            captured_prints.append(" ".join(str(a) for a in args))
            _real_print(*args,**kwargs)  # also print to terminal so you can watch
        env=self._get_sandbox()
        env["print"]=capture_print
        env["inputs"]=inputs or{}
        env["result"]=None
        try:
            exec(code,env)
            result=env.get("result")
            output_parts=[]
            if captured_prints: output_parts.append("\n".join(captured_prints))
            if result is not None: output_parts.append(f"result = {result}")
            output="\n".join(output_parts) if output_parts else "(no output)"
            return{"success":True,"output":output,"result":result,"prints":captured_prints,"error":None}
        except Exception as ex:
            return{"success":False,"output":None,"result":None,"prints":captured_prints,"error":str(ex)}

    # ── heartbeat observations ───────────────────────────────────────────
    def gather_observations(self):
        conv_count=len(self.nodes_of_type(CONVERSATION))
        return{"node_count":len(self.nodes),"edge_count":len(self.edges),
            "avg_relevance":round(sum(n.relevance_score for n in self.nodes.values())/max(1,len(self.nodes)),3),
            "near_archive":sum(1 for n in self.nodes.values() if n.relevance_score<0.12 and n.type in(CONVERSATION,META)),
            "fatigue":round(self.metabolism.fatigue,3),
            "energy_pct":round(self.metabolism.energy_remaining/max(1,self.metabolism.total_energy),3),
            "recovery":round(self.metabolism.recovery_momentum,3),
            "debt":round(self.metabolism.resource_debt,2),
            "error_rate":round(self.self_model.recent_error_rate,3),
            "queries_epoch":self.metabolism.queries_this_epoch,
            "drives":self.drive_state(),"tool_count":len(self.get_tools()),
            "tools_low_success":[t.id for t in self.get_tools() if t.success_rate<0.5 and t.total_executions>3],
            "quarantined":sum(1 for e in self.edges.values() if e.quarantined),
            "tick":self.tick_count,"conv_count":conv_count,
            "cognitive_count":len(self.get_cognitive_nodes(n=100)),
            "recent_thoughts":[t.get("summary","") for t in self.heartbeat.thought_history[-3:]]}

    def should_synthesize(self):
        conv_count=len(self.nodes_of_type(CONVERSATION))
        return conv_count-self.heartbeat.last_synthesis_conv_count>=self.config.get("synthesis_min_new_convs")

    def store_thought(self,thought_type,content,actions=None):
        n=Node(META,content,extract_kw(content),emotional_salience=0.15,
               meta={"thought_type":thought_type,"tick":self.tick_count,"actions_taken":actions or[]})
        n.thought_type=thought_type;n.tick=self.tick_count;n.actions_taken=actions or[]
        n.decay_rate=0.990;self.add_node(n)
        prev=self.get_thoughts(2)
        if len(prev)>1:
            self.add_edge(Edge(n.id,prev[1].id,channels={"temporal":0.9,"cognitive_sequence":0.8},
                confidence=0.7,label="follows_thought"))
        self.heartbeat.record(thought_type,content,actions or[]);return n

    # ── cognitive synthesis with dedup + cross-linking ────────────────────
    def find_similar_cognitive(self,content,threshold=None):
        threshold=threshold or self.config.get("cognitive_dedup_threshold")
        matches=[]
        for n in self.nodes.values():
            if n.type not in(INSIGHT,THEORY,STRATEGY,GOAL): continue
            sim=text_similarity(content,n.content)
            if sim>=threshold: matches.append((sim,n))
        matches.sort(key=lambda x:-x[0]);return matches

    def store_cognitive(self,node_type,content,source_ids=None,confidence=0.5):
        """Unified: dedup check, create if novel, cross-link to related."""
        similar=self.find_similar_cognitive(content)
        if similar:
            _,existing=similar[0]
            existing.touch();existing.relevance_score=min(1.0,existing.relevance_score+0.1)
            for sid in(source_ids or[]):
                if not any(e.source_id==existing.id and e.target_id==sid for e in self.edges.values()):
                    if self.get(sid):
                        self.add_edge(Edge(existing.id,sid,channels={"reinforced_by":0.6},
                            confidence=confidence,label="reinforced"))
            return None  # deduped
        decay={INSIGHT:0.996,THEORY:0.994,STRATEGY:0.996,GOAL:0.998}
        sal={INSIGHT:0.5,THEORY:0.6,STRATEGY:0.5,GOAL:0.6}
        res={INSIGHT:0.15,THEORY:0.2,STRATEGY:0.25,GOAL:0.3}
        n=Node(node_type,content,extract_kw(content),emotional_salience=sal.get(node_type,0.5),
               resistance=res.get(node_type,0.15))
        n.decay_rate=decay.get(node_type,0.996)
        if node_type==GOAL: n.meta["priority"]=confidence
        self.add_node(n)
        elbl={INSIGHT:"insight_from",THEORY:"evidence",STRATEGY:"strategy_for",GOAL:"goal_for"}
        for sid in(source_ids or[]):
            if self.get(sid):
                self.add_edge(Edge(n.id,sid,channels={"derived_from":0.7},
                    confidence=confidence,provenance=INFERRED,label=elbl.get(node_type,"related")))
        # cross-link to related cognitive nodes
        related=self.find_similar_cognitive(content,threshold=0.25)
        for sim,other in related[:3]:
            if other.id==n.id: continue
            self.add_edge(Edge(n.id,other.id,channels={"related_to":round(sim,2)},
                confidence=round(sim,2),provenance=INFERRED,label="related_cognition"))
        if node_type==GOAL:
            for idn in self.nodes_of_type(IDENTITY):
                if idn.id=="mission":
                    self.add_edge(Edge(n.id,idn.id,channels={"aligned_with":0.6},
                        confidence=0.5,label="serves_mission"));break
        return n

    def store_insight(self,c,s=None,conf=0.6): return self.store_cognitive(INSIGHT,c,s,conf)
    def store_theory(self,c,s=None,conf=0.4): return self.store_cognitive(THEORY,c,s,conf)
    def store_strategy(self,c,s=None,conf=0.6): return self.store_cognitive(STRATEGY,c,s,conf)
    def store_goal(self,c,priority=0.5,p=None): return self.store_cognitive(GOAL,c,p,priority)

    def get_cognitive_nodes(self,types=None,n=20):
        target=types or{INSIGHT,THEORY,STRATEGY,GOAL}
        nodes=[nd for nd in self.nodes.values() if nd.type in target]
        nodes.sort(key=lambda x:x.creation_time,reverse=True);return nodes[:n]

    # ── graph surgery ────────────────────────────────────────────────────
    def merge_nodes(self,id_a,id_b):
        a,b=self.get(id_a),self.get(id_b)
        if not a or not b: return None
        if a.type in(IDENTITY,DRIVE) or b.type in(IDENTITY,DRIVE): return None
        keeper,donor=(a,b) if a.relevance_score>=b.relevance_score else(b,a)
        keeper.content=keeper.content+" | "+donor.content
        keeper.keywords=list(set(keeper.keywords+donor.keywords))[:15]
        keeper.relevance_score=max(keeper.relevance_score,donor.relevance_score)
        for e in list(self.edges.values()):
            if e.source_id==donor.id: e.source_id=keeper.id
            if e.target_id==donor.id: e.target_id=keeper.id
        self.remove_node(donor.id);keeper.touch();return keeper

    # ── dreaming (graph consolidation + theory formation + router decay) ──
    def dream_cycle(self):
        """Offline consolidation: merge dupes, prune, theory formation, router decay."""
        actions_taken=[]

        # 1. Merge similar cognitive nodes
        cognitive=self.get_cognitive_nodes(n=50)
        merged_ids=set()
        for i,a in enumerate(cognitive):
            if a.id in merged_ids: continue
            for b in cognitive[i+1:]:
                if b.id in merged_ids: continue
                sim=text_similarity(a.content,b.content)
                if sim>0.6:
                    result=self.merge_nodes(a.id,b.id)
                    if result:
                        merged_ids.add(a.id if result.id!=a.id else b.id)
                        actions_taken.append(f"merged cognitive: {a.label(30)} + {b.label(30)}")

        # 2. Merge similar META (thought) nodes
        thoughts=[n for n in self.nodes.values() if n.type==META]
        thoughts.sort(key=lambda x:x.creation_time,reverse=True)
        merged_ids2=set()
        for i,a in enumerate(thoughts[:30]):
            if a.id in merged_ids2: continue
            for b in thoughts[i+1:i+8]:
                if b.id in merged_ids2: continue
                sim=text_similarity(a.content,b.content)
                if sim>0.5:
                    result=self.merge_nodes(a.id,b.id)
                    if result:
                        merged_ids2.add(a.id if result.id!=a.id else b.id)
                        actions_taken.append(f"merged thoughts: {a.label(25)} + {b.label(25)}")

        # 3. Prune orphan nodes (no edges, low relevance, not protected types)
        protected={IDENTITY,DRIVE,TOOL,GOAL}
        pruned=0
        for n in list(self.nodes.values()):
            if n.type in protected: continue
            if len(self.edges_of(n.id))==0 and n.relevance_score<0.15:
                self.archive[n.id]=n.to_dict()
                self.remove_node(n.id)
                pruned+=1
        if pruned: actions_taken.append(f"pruned {pruned} orphan nodes")

        # 4. Remove dead edges (both endpoints low relevance)
        dead_edges=[]
        for e in list(self.edges.values()):
            src=self.get(e.source_id)
            tgt=self.get(e.target_id)
            if not src or not tgt:
                dead_edges.append(e.id)
            elif src.relevance_score<0.05 and tgt.relevance_score<0.05:
                dead_edges.append(e.id)
        for eid in dead_edges:
            self.edges.pop(eid,None)
        if dead_edges: actions_taken.append(f"removed {len(dead_edges)} dead edges")

        # 5. Reinforce frequently-used edges
        reinforced=0
        for e in self.edges.values():
            if e.usage_count>3:
                e.confidence=min(1.0,e.confidence*1.05)
                e.usage_count=0
                reinforced+=1
        if reinforced: actions_taken.append(f"reinforced {reinforced} hot edges")

        # 6. Remove self-loops
        self_loops=[eid for eid,e in self.edges.items() if e.source_id==e.target_id]
        for eid in self_loops: self.edges.pop(eid,None)
        if self_loops: actions_taken.append(f"removed {len(self_loops)} self-loops")

        # 7. Router decay — all router scores pull toward 0.5 (neutral)
        router_decayed=0
        for n in self.nodes.values():
            if n.router:
                for ch in list(n.router.keys()):
                    old=n.router[ch]
                    n.router[ch]=old*0.995+0.5*0.005  # decay toward 0.5
                router_decayed+=1
        if router_decayed: actions_taken.append(f"decayed routers on {router_decayed} nodes")

        # 8. Theory formation — find patterns across game outcomes
        theories_formed=self._form_theories()
        if theories_formed: actions_taken.extend(theories_formed)

        # 9. Theory confidence decay — pull toward 0.5 (max uncertainty)
        for n in self.get_cognitive_nodes(types={THEORY},n=50):
            ec=n.meta.get("evidence_count",0)
            if ec>0:
                n.meta["confidence"]=n.meta.get("confidence",0.5)*0.98+0.5*0.02

        # 10. Theory pruning — deactivate failed theories
        pruned_theories=0
        for n in list(self.get_cognitive_nodes(types={THEORY},n=50)):
            ec=n.meta.get("evidence_count",0)
            conf=n.meta.get("confirmation_rate",0.5)
            if ec>=3 and conf<0.25:
                n.meta["active"]=False
                n.relevance_score=0.01
                pruned_theories+=1
        if pruned_theories: actions_taken.append(f"pruned {pruned_theories} failed theories")

        return actions_taken

    def _form_theories(self):
        """Find patterns in conversations that could become theories.
        Look for: common keywords in winning vs losing game conversations."""
        actions=[]
        convs=self.get_conversations(30)
        if len(convs)<6: return actions

        # Separate winning and losing conversations
        win_kw=defaultdict(int);lose_kw=defaultdict(int)
        for c in convs:
            is_win="WON" in c.content.upper()
            is_lose="LOST" in c.content.upper()
            words=set(re.findall(r'\w{4,}',c.content.lower()))
            target=win_kw if is_win else lose_kw if is_lose else None
            if target:
                for w in words: target[w]+=1

        if not win_kw or not lose_kw: return actions

        # Find keywords that appear significantly more in wins than losses
        all_kw=set(win_kw.keys())|set(lose_kw.keys())
        stop={'move','board','score','black','white','game','played','captures','assistant','user'}
        patterns=[]
        for kw in all_kw:
            if kw in stop: continue
            wc=win_kw.get(kw,0);lc=lose_kw.get(kw,0)
            total=wc+lc
            if total<3: continue
            win_rate=wc/total
            if win_rate>0.7: patterns.append((kw,"positive",win_rate,total))
            elif win_rate<0.3: patterns.append((kw,"negative",win_rate,total))

        # Create theory nodes for strong patterns (max 2 per dream)
        for kw,direction,rate,evidence in sorted(patterns,key=lambda x:-x[3])[:2]:
            if direction=="positive":
                content=f"Positions involving '{kw}' tend to lead to wins ({rate:.0%} win rate over {evidence} games)"
            else:
                content=f"Positions involving '{kw}' tend to lead to losses ({rate:.0%} win rate over {evidence} games)"
            # Check if we already have this theory
            existing=self.find_similar_cognitive(content)
            if existing:
                _,e=existing[0]
                e.meta["evidence_count"]=e.meta.get("evidence_count",0)+evidence
                e.meta["confirmation_rate"]=rate
                e.touch()
                continue
            n=self.store_cognitive(THEORY,content)
            if n:
                n.meta["conditions"]=[kw]
                n.meta["direction"]=direction
                n.meta["confirmation_rate"]=rate
                n.meta["evidence_count"]=evidence
                n.meta["active"]=True
                actions.append(f"theory: '{kw}' → {direction} ({rate:.0%}, n={evidence})")
        return actions

    # ── persistence ──────────────────────────────────────────────────────
    def save(self):
        data={"nodes":[n.to_dict() for n in self.nodes.values()],
              "edges":[e.to_dict() for e in self.edges.values()],
              "archive":dict(list(self.archive.items())[-200:]),  # keep last 200 archived
              "archive_count":len(self.archive),"metabolism":self.metabolism.to_dict(),
              "self_model":self.self_model.to_dict(),"heartbeat":self.heartbeat.to_dict(),
              "config":self.config.to_dict(),"tick_count":self.tick_count}
        os.makedirs(os.path.dirname(self.SAVE_PATH),exist_ok=True)
        tmp=self.SAVE_PATH+".tmp"
        with open(tmp,"w") as f: json.dump(data,f,separators=(",",":"))
        os.replace(tmp,self.SAVE_PATH)

    def load(self):
        if not os.path.exists(self.SAVE_PATH): return False
        with open(self.SAVE_PATH) as f: data=json.load(f)
        self.nodes={d["id"]:Node.from_dict(d) for d in data.get("nodes",[])}
        self.edges={d["id"]:Edge.from_dict(d) for d in data.get("edges",[])}
        self.archive=data.get("archive",{})
        if "metabolism" in data: self.metabolism=MetabolicState.from_dict(data["metabolism"])
        if "self_model" in data: self.self_model=SelfModel.from_dict(data["self_model"])
        if "heartbeat" in data: self.heartbeat=HeartbeatState.from_dict(data["heartbeat"])
        if "config" in data: self.config=Config.from_dict(data["config"])
        self.tick_count=data.get("tick_count",0);return True

    # ── diagnostics ──────────────────────────────────────────────────────
    def diagnostics(self):
        tc=defaultdict(int)
        for n in self.nodes.values(): tc[n.type]+=1
        return{"node_count":len(self.nodes),
            "edge_count":sum(1 for e in self.edges.values() if not e.quarantined),
            "quarantined_edges":sum(1 for e in self.edges.values() if e.quarantined),
            "archived_count":len(self.archive),"type_counts":dict(tc),
            "avg_relevance":round(sum(n.relevance_score for n in self.nodes.values())/max(1,len(self.nodes)),4),
            "tick_count":self.tick_count,"metabolism":self.metabolism.to_dict(),
            "self_model":self.self_model.to_dict(),"heartbeat":self.heartbeat.to_dict(),
            "config":self.config.to_dict(),
            "identity":[{"id":n.id,"content":n.content,"resistance":n.resistance,
                "pressure":round(n.pressure_accumulator,4),"modification_log":n.modification_log[-3:]}
                for n in self.nodes_of_type(IDENTITY)],
            "drives":[{"id":n.id,"intensity":round(n.intensity,3),"baseline":round(n.baseline_intensity,3),
                "content":n.content,"exploration_weight":n.meta.get("exploration_weight",0)}
                for n in self.nodes_of_type(DRIVE)],
            "tools":[{"id":n.id,"name":n.keywords[0] if n.keywords else n.id,
                "description":n.label(50),"success_rate":round(n.success_rate,3),
                "executions":n.total_executions,"version":n.version,"risk":n.risk_level}
                for n in self.nodes_of_type(TOOL)],
            "cognition":[{"id":n.id,"type":n.type,"content":n.content[:200],
                "relevance":round(n.relevance_score,3),"age_hours":round(n.age_hours(),1),
                "edge_count":len(self.edges_of(n.id))} for n in self.get_cognitive_nodes(n=10)]}

    def viz_data(self,full=False):
        always={IDENTITY,DRIVE,TOOL,INSIGHT,THEORY,STRATEGY,GOAL}
        nodes_out=[]
        for n in self.nodes.values():
            if not full and n.relevance_score<0.04 and n.type not in always: continue
            d={"id":n.id,"type":n.type,"label":n.label(28),"content":n.content[:250],
               "relevance":round(n.relevance_score,3),"access_count":n.access_count,
               "keywords":n.keywords[:6],"age_hours":round(n.age_hours(),1)}
            if n.type==IDENTITY: d["resistance"]=n.resistance
            if n.type==DRIVE: d["intensity"]=round(n.intensity,3)
            if n.type==TOOL: d["success_rate"]=round(n.success_rate,3);d["executions"]=n.total_executions;d["risk"]=n.risk_level
            if n.type==META: d["thought_type"]=n.thought_type
            if n.type==GOAL: d["priority"]=n.meta.get("priority",0.5)
            nodes_out.append(d)
        nids={n["id"] for n in nodes_out};edges_out=[]
        for e in self.edges.values():
            if e.source_id in nids and e.target_id in nids:
                if e.quarantined and not full: continue
                edges_out.append({"id":e.id,"source":e.source_id,"target":e.target_id,
                    "channels":e.channels,"confidence":round(e.confidence,3),
                    "provenance":e.provenance,"label":e.label,
                    "quarantined":e.quarantined,"usage_count":e.usage_count})
        return{"nodes":nodes_out,"edges":edges_out}