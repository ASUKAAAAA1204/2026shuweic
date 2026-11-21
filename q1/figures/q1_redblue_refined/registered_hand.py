"""Anatomical landmarks on the generated palmar illustration, in its portrait coordinates."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Polygon
import numpy as np
from scipy.interpolate import LinearNDInterpolator


def make_hand(image,xyz,previous,names):
    # Image-plane anatomical correspondences on a 510-pixel-wide reference view.
    template={
        "Wrist": [(253,526)],
        "Thumb": [(148,405),(113,333),(80,285),(38,218)],
        "Index": [(189,239),(177,159),(185,111),(201,53)],
        "Middle": [(261,229),(261,146),(281,85),(299,20)],
        "Ring": [(327,244),(335,180),(349,130),(363,76)],
        "Little": [(387,293),(416,237),(418,204),(399,166)],
    }
    image_coords={"rightWrist":template["Wrist"][0]}
    for finger,locations in list(template.items())[1:]:
        for joint,xy in zip(("MCP","PIP","DIP","Tip"),locations):
            image_coords[f"right{finger}{joint}"]=xy
    assert len(names)==len(set(names))==21 and set(names)==set(image_coords)
    target=np.array([image_coords[n] for n in names],float)
    wrist=xyz[names.index("rightWrist")]
    def unit(v):
        return v/np.linalg.norm(v)
    up=unit(xyz[names.index("rightMiddleMCP")]-wrist)
    lateral=xyz[names.index("rightLittleMCP")]-xyz[names.index("rightIndexMCP")]
    right=unit(lateral-up*np.dot(up,lateral))
    normal=unit(np.cross(right,up))
    view=unit(-.4*right+.55*up+normal)
    horiz=unit(right-view*np.dot(view,right))
    basis=np.stack([horiz,np.cross(view,horiz)],axis=1)
    source=(xyz-wrist)@basis
    register=LinearNDInterpolator(source,target)
    np.testing.assert_allclose(register(source),target,atol=1e-7)
    centers=(np.stack([previous.mean(0),xyz.mean(0)])-wrist)@basis
    before,now=register(centers)
    assert np.isfinite([before,now]).all()
    displacement=3.5*(now-before)
    perpendicular=np.array([-displacement[1],displacement[0]])
    # Tail, neck, shoulder and tip in units of the full arrow length.
    dart=np.array([(0,0),(.77,-.032),(.77,-.095),(1,0),(.77,.095),(.77,.032)])
    vertices=now+dart[:,:1]*displacement+dart[:,1:]*perpendicular

    fig,ax=plt.subplots(figsize=(5.25,5.1))
    bitmap=plt.imread(image)
    ax.imshow(bitmap,extent=(0,510,510*bitmap.shape[0]/bitmap.shape[1],0))
    for finger in list(template)[1:]:
        ids=[names.index("rightWrist")]+[names.index("right"+finger+j) for j in ("MCP","PIP","DIP","Tip")]
        ax.plot(*target[ids].T,color="#234E76",lw=1.25,alpha=.84,zorder=2)
    ax.scatter(*target.T,s=28,facecolor="#2A73BA",edgecolor="white",linewidth=1,zorder=4)
    ax.scatter(*before,s=50,facecolor="white",edgecolor="#CE4844",linewidth=1.5,zorder=5)
    ax.scatter(*now,s=46,c="#1F4E89",marker="D",edgecolor="white",lw=.8,zorder=6)
    ax.add_patch(Polygon(vertices,closed=True,facecolor="#FF0000",edgecolor="none",zorder=7))
    ax.axis("off")
    handles=[Line2D([],[],linestyle="",marker=m,markerfacecolor=c,markeredgecolor=e,label=n)
             for m,c,e,n in [("o","#2A73BA","white","关键点（21 个）"),("o","white","#CE4844","前帧质心"),
                             ("D","#1F4E89","white","当前质心")]]
    # Marker coordinates have an upward y-axis, unlike the illustration.
    symbol=(vertices-(now+displacement/2))*[1,-1]
    handles.append(Line2D([],[],linestyle="",marker=Path(np.vstack([symbol,symbol[0]])),
                          markersize=13,markerfacecolor="#FF0000",markeredgewidth=0,label="初速方向"))
    ax.legend(handles=handles,loc="center left",bbox_to_anchor=(.91,.12),frameon=False,fontsize=9)
    return fig
