#!/usr/bin/env python3
"""Write animation using EXACT T_full from weight transfer.

T_full = P4 @ T_align_wt
where T_align_wt = _aabb_align_uniform(soma_verts, raw_trellis_verts)
and P4 is the Z-up → Y-up conversion.
"""
import numpy as np
import json
import struct
import math
import sys
sys.path.insert(0, '/root/ComfyUI/custom_nodes/melite-autorig-nodes')
from soma_weight_transfer import _aabb_align_uniform, _read_glb, SOMA_SKIN_PATH

SOMA77_PARENTS = (
    -1,0,1,2,3,4,5,6,6,6,6,3,11,12,13,14,15,16,17,14,19,20,21,22,
    14,24,25,26,27,14,29,30,31,32,14,34,35,36,37,3,39,40,41,42,
    43,44,45,42,47,48,49,50,42,52,53,54,55,42,57,58,59,60,42,62,
    63,64,65,0,67,68,69,70,0,72,73,74,75,
)
SOMA_TO_MIXAMO = {
    0:"Hips",1:"Spine",2:"Spine1",3:"Spine2",4:"Neck",5:"Neck1",6:"Head",
    7:"HeadTop_End",8:"Jaw",9:"LeftEye",10:"RightEye",
    11:"LeftShoulder",12:"LeftArm",13:"LeftForeArm",14:"LeftHand",
    15:"LeftHandThumb1",16:"LeftHandThumb2",17:"LeftHandThumb3",18:"LeftHandThumb4",
    19:"LeftHandIndex1",20:"LeftHandIndex2",21:"LeftHandIndex3",22:"LeftHandIndex4",
    24:"LeftHandMiddle1",25:"LeftHandMiddle2",26:"LeftHandMiddle3",27:"LeftHandMiddle4",
    29:"LeftHandRing1",30:"LeftHandRing2",31:"LeftHandRing3",32:"LeftHandRing4",
    34:"LeftHandPinky1",35:"LeftHandPinky2",36:"LeftHandPinky3",37:"LeftHandPinky4",
    39:"RightShoulder",40:"RightArm",41:"RightForeArm",42:"RightHand",
    43:"RightHandThumb1",44:"RightHandThumb2",45:"RightHandThumb3",46:"RightHandThumb4",
    47:"RightHandIndex1",48:"RightHandIndex2",49:"RightHandIndex3",50:"RightHandIndex4",
    52:"RightHandMiddle1",53:"RightHandMiddle2",54:"RightHandMiddle3",55:"RightHandMiddle4",
    57:"RightHandRing1",58:"RightHandRing2",59:"RightHandRing3",60:"RightHandRing4",
    62:"RightHandPinky1",63:"RightHandPinky2",64:"RightHandPinky3",65:"RightHandPinky4",
    67:"LeftUpLeg",68:"LeftLeg",69:"LeftFoot",70:"LeftToeBase",71:"LeftToe_End",
    72:"RightUpLeg",73:"RightLeg",74:"RightFoot",75:"RightToeBase",76:"RightToe_End",
}

def quat_to_mat(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]], dtype=np.float64)

def trs_to_mat(T,R,S):
    T=T or [0,0,0];R=R or [0,0,0,1];S=S or [1,1,1]
    m=np.eye(4,dtype=np.float64);m[:3,:3]=quat_to_mat(R)*np.array(S);m[:3,3]=T;return m

def mat4(rot,trans):
    m=np.eye(4,dtype=np.float64);m[:3,:3]=rot;m[:3,3]=trans;return m

def rot_to_quat(R):
    tr=R[0,0]+R[1,1]+R[2,2]
    if tr>0:s=0.5/math.sqrt(tr+1);w=0.25/s;x=(R[2,1]-R[1,2])*s;y=(R[0,2]-R[2,0])*s;z=(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:s=2*math.sqrt(1+R[0,0]-R[1,1]-R[2,2]);w=(R[2,1]-R[1,2])/s;x=0.25*s;y=(R[0,1]+R[1,0])/s;z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:s=2*math.sqrt(1+R[1,1]-R[0,0]-R[2,2]);w=(R[0,2]-R[2,0])/s;x=(R[0,1]+R[1,0])/s;y=0.25*s;z=(R[1,2]+R[2,1])/s
    else:s=2*math.sqrt(1+R[2,2]-R[0,0]-R[1,1]);w=(R[1,0]-R[0,1])/s;x=(R[0,2]+R[2,0])/s;y=(R[1,2]+R[2,1])/s;z=0.25*s
    q=np.array([x,y,z,w],dtype=np.float64);n=np.linalg.norm(q)
    if n>0:q/=n
    if q[3]<0:q=-q
    return q

def main():
    rigged_glb=sys.argv[1];npz_path=sys.argv[2];output=sys.argv[3]
    fps=int(sys.argv[4]) if len(sys.argv)>4 else 30

    motion=np.load(npz_path,allow_pickle=True)
    posed_joints=motion["posed_joints"].astype(np.float64)
    global_rots=motion["global_rot_mats"].astype(np.float64)
    T_frames,n_joints=posed_joints.shape[:2]

    soma=np.load(SOMA_SKIN_PATH,allow_pickle=True)
    soma_verts=soma["bind_vertices"].astype(np.float32)
    bind_rig=soma["bind_rig_transform"].astype(np.float64)

    # Read raw TRELLIS GLB to get source vertices
    raw_gltf, raw_bin = _read_glb("/tmp/trellis_test.glb")
    prim = raw_gltf["meshes"][0]["primitives"][0]
    pos_acc = raw_gltf["accessors"][prim["attributes"]["POSITION"]]
    pos_bv = raw_gltf["bufferViews"][pos_acc["bufferView"]]
    pos_off = pos_bv.get("byteOffset",0) + pos_acc.get("byteOffset",0)
    raw_verts = np.frombuffer(raw_bin, dtype=np.float32, count=pos_acc["count"]*3, offset=pos_off).reshape(pos_acc["count"],3).astype(np.float32)

    # Compute EXACT T_align (same as weight transfer)
    T_align_wt = _aabb_align_uniform(soma_verts, raw_verts)
    
    # P4: Z-up → Y-up
    P4 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64)
    
    # T_full = P4 @ T_align_wt
    T_full = P4 @ T_align_wt.astype(np.float64)
    T_full_inv = np.linalg.inv(T_full)
    
    scale = abs(np.linalg.det(T_full[:3,:3]))**(1/3)
    print(f"T_full exact scale: {scale:.4f}")

    # Read rigged GLB
    with open(rigged_glb,'rb') as f:
        f.read(12)
        jl,_=struct.unpack('<II',f.read(8))
        gltf=json.loads(f.read(jl).decode())
        bl,_=struct.unpack('<II',f.read(8))
        orig_bin=bytearray(f.read(bl))

    nodes=gltf["nodes"]
    jnl=gltf["skins"][0]["joints"]

    soma_to_jli={}
    for si,mn in SOMA_TO_MIXAMO.items():
        if si>=n_joints:continue
        for jli,jn in enumerate(jnl):
            nm=nodes[jn].get('name','').replace('mixamorig:','')
            if nm==mn:soma_to_jli[si]=jli;break
    print(f"Mapped {len(soma_to_jli)}/77 joints")

    # Rest global
    gp={}
    for ni,node in enumerate(nodes):
        for c in node.get("children",[]):gp[c]=ni
    rg={}
    def grg(ni):
        if ni in rg:return rg[ni]
        node=nodes[ni];l=trs_to_mat(node.get("translation"),node.get("rotation"),node.get("scale"))
        p=gp.get(ni,-1);rg[ni]=grg(p)@l if p>=0 else l;return rg[ni]
    for jn in jnl:grg(jn)

    # Compute D_glb = T_full @ D_soma @ T_full^-1
    D_glb_all={}
    for i in range(n_joints):
        dl=[]
        for t in range(T_frames):
            G_t=mat4(global_rots[t,i],posed_joints[t,i])
            D_soma=G_t@np.linalg.inv(bind_rig[i])
            D_glb=T_full@D_soma@T_full_inv
            dl.append(D_glb)
        D_glb_all[i]=dl

    # nodeGlobal = D_glb @ rest_global
    ng_all={}
    for si in soma_to_jli:
        jn=jnl[soma_to_jli[si]];r=rg[jn]
        ng_all[si]=[D_glb_all[si][t]@r for t in range(T_frames)]

    # Local transforms
    at={};ar={}
    for si in soma_to_jli:
        ps=SOMA77_PARENTS[si];jn=jnl[soma_to_jli[si]];g=gp.get(jn,-1)
        tl=np.zeros((T_frames,3),dtype=np.float32);rl=np.zeros((T_frames,4),dtype=np.float32)
        for t in range(T_frames):
            n=ng_all[si][t]
            if ps>=0 and ps in soma_to_jli:n_p=ng_all[ps][t];l=np.linalg.inv(n_p)@n
            elif g>=0:l=np.linalg.inv(rg[g])@n
            else:l=n
            tl[t]=l[:3,3].astype(np.float32);rl[t]=rot_to_quat(l[:3,:3]).astype(np.float32)
        at[si]=tl;ar[si]=rl

    # Build GLB
    nb=bytearray(orig_bin)
    while len(nb)%4:nb.append(0)
    as_=len(nb)
    tv=(np.arange(T_frames)/fps).astype(np.float32)
    ad=bytearray()
    to=len(ad);ad.extend(tv.tobytes())
    tro={};roo={}
    for si in sorted(soma_to_jli):
        tro[si]=len(ad);ad.extend(at[si].tobytes())
        roo[si]=len(ad);ad.extend(ar[si].tobytes())
    while len(ad)%4:ad.append(0)
    nb.extend(ad)

    gltf["animations"]=[]
    bvs=gltf.setdefault("bufferViews",[])
    accs=gltf.setdefault("accessors",[])
    tbv=len(bvs);bvs.append({"buffer":0,"byteOffset":as_+to,"byteLength":T_frames*4})
    tac=len(accs);accs.append({"bufferView":tbv,"componentType":5126,"count":T_frames,"min":[float(tv[0])],"max":[float(tv[-1])],"type":"SCALAR"})
    sam=[];ch=[]
    for si in sorted(soma_to_jli):
        ni=jnl[soma_to_jli[si]]
        k=len(bvs);bvs.append({"buffer":0,"byteOffset":as_+tro[si],"byteLength":T_frames*12})
        ka=len(accs);accs.append({"bufferView":k,"componentType":5126,"count":T_frames,"min":at[si].min(0).tolist(),"max":at[si].max(0).tolist(),"type":"VEC3"})
        st=len(sam);sam.append({"input":tac,"output":ka});ch.append({"sampler":st,"target":{"node":ni,"path":"translation"}})
        k2=len(bvs);bvs.append({"buffer":0,"byteOffset":as_+roo[si],"byteLength":T_frames*16})
        ka2=len(accs);accs.append({"bufferView":k2,"componentType":5126,"count":T_frames,"type":"VEC4"})
        st2=len(sam);sam.append({"input":tac,"output":ka2});ch.append({"sampler":st2,"target":{"node":ni,"path":"rotation"}})
    gltf["animations"]=[{"name":"SOMAWalk","samplers":sam,"channels":ch}]
    gltf["buffers"][0]["byteLength"]=len(nb)
    js=json.dumps(gltf,separators=(',',':')).encode()
    while len(js)%4:js+=b' '
    total=12+8+len(js)+8+len(nb)
    with open(output,'wb') as f:
        f.write(struct.pack('<III',0x46546C67,2,total))
        f.write(struct.pack('<II',len(js),0x4E4F534A));f.write(js)
        f.write(struct.pack('<II',len(nb),0x004E4942));f.write(nb)
    print(f"Written: {output} ({total/1024:.0f}KB)")

if __name__=="__main__":
    main()
