import lietorch
import torch
import torch.nn.functional as F

from .chol import block_solve, schur_solve
import geom.projective_ops as pops

from torch_scatter import scatter_sum

#TODO(PARTH) SORT OUT THE THESEUS ISSUE
# import theseus as th
# from theseus.core import Variable
# from theseus.optimizer import Optimizer
# from theseus.geometry import SO3
# # from theseus.optim import LM

# #print(dir(th))
# class projectTrans(th.CostFunction):
#     def __init__(self, name, poses, depths, intrinsics, ii, jj, target, wgt, eta, original_depths_shape, original_poses_shape,cost_weight=th.ScaleCostWeight(1.0)):
#         #cost_weight = 
#         th.CostFunction.__init__(self, name=name, cost_weight=cost_weight)

#         # Store all Theseus variables
#         self.poses = poses
#         self.depths = depths
#         self.intrinsics = intrinsics
#         self.ii = ii
#         self.jj = jj
#         self.target = target
#         self.wgt = wgt
#         self.eta = eta
#         self.original_depths_shape = original_depths_shape
#         self.original_poses_shape = original_poses_shape
      
#         self.register_optim_vars(["poses", "depths"])
#         self.register_aux_vars(["intrinsics", "ii", "jj", "target", "wgt", "eta", "original_depths_shape", "original_poses_shape"])
    
#     def error(self) -> torch.Tensor:
#         # Extract shapes from auxiliary variables
#         #print(self.original_poses_shape.tensor[0].tolist())
#         #print(self.original_depths_shape.tensor[0].tolist())
#         poses_shape = torch.Size(self.original_poses_shape.tensor[0].to(torch.int32).tolist())
#         depths_shape = torch.Size(self.original_depths_shape.tensor[0].to(torch.int32).tolist())

#         poses_tensor = self.poses.tensor.view(poses_shape)
#         depths_tensor = self.depths.tensor.view(depths_shape)
#         intriniscs = self.intrinsics.tensor.view(-1,4).squeeze(0)
#         #print("ii",self.ii)
#         #print(self.intrinsics,self.intrinsics.tensor.shape)
#         #print("Intriniscs shape",intriniscs.shape)
        
#         coords, valid = pops.projective_transform(
#             poses_tensor,
#             depths_tensor,
#             intriniscs,
#             self.ii.tensor.to(torch.long),
#             self.jj.tensor.to(torch.long)
#         )
#         r = (self.target.tensor - coords).view(-1, 2)
#         w = 0.001 * (valid * self.weight.tensor).view(-1, 2)
#         #TODO(SOLVE THE 90 GB MEM ISSUE)
#         # return w * r
#         return coords -self.target.tensor
    


#     def jacobians(self):
#         _, _, (Ji, Jj, Jz) = pops.projective_transform(
#             self.poses.tensor[None],
#             self.depths.tensor[None],
#             self.intrinsics.tensor[None],
#             self.ii.tensor.to(torch.long),
#             self.jj.tensor.to(torch.long),
#             jacobian=True
#         )
#         return [Ji, Jj, Jz] , self.error()
#         return [] , self.error()
    


#     def _copy_impl(self, new_name=None)->"projectTrans":
#         return projectTrans(
#             name=new_name or self.name,
#             poses=self.poses.copy(),
#             depths=self.depths.copy(),
#             intrinsics=self.intrinsics.copy(),
#             ii=self.ii.copy(),
#             jj=self.jj.copy(),
#             target=self.target.copy(),
#             wgt=self.wgt.copy(),
#             eta=self.eta.copy(),
#             original_depths_shape=self.original_depths_shape.copy(),
#             original_poses_shape=self.original_poses_shape.copy()
#         )

#     def dim(self) -> int:
#         # The dimension of the residual is 2 per measurement
#         return 2

def BA_theseus(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    # Convert to float32
    poses = poses.float()
    disps = disps.float()
    intrinsics = intrinsics.float()
    target = target.float()
    weight = weight.float()
    ii = ii.float()
    jj = jj.float()
    eta = eta.float()

    # Theseus variables
    poses_var = th.Vector(tensor=poses.view(1, -1).to("cpu"), name="poses")
    disps_var = th.Vector(tensor=disps.view(1, -1).to("cpu"), name="depths")
    intrinsics_var = th.Vector(tensor=intrinsics.view(1, -1).to("cpu"), name="intrinsics")
    target_var = th.Vector(tensor=target.view(1, -1).to("cpu"), name="target")
    weight_var = th.Vector(tensor=weight.view(1, -1).to("cpu"), name="wgt")
    ii_var = th.Vector(tensor=ii.view(1, -1).to("cpu"), name="ii")
    jj_var = th.Vector(tensor=jj.view(1, -1).to("cpu"), name="jj")
    eta_var = th.Vector(tensor=eta.view(1, -1).to("cpu"), name="eta")
    original_depths_shape = th.Vector(tensor=torch.as_tensor(disps.shape, dtype=torch.float32).view(1, -1).to("cpu"), name="original_depths_shape")
    original_poses_shape = th.Vector(tensor=torch.as_tensor(poses.shape, dtype=torch.float32).view(1, -1).to("cpu"), name="original_poses_shape")



    # Cost function
    cost_fn = projectTrans(
        name="PT",
        poses=poses_var,
        depths=disps_var,
        intrinsics=intrinsics_var,
        target=target_var,
        wgt=weight_var,
        ii=ii_var,
        jj=jj_var,
        eta=eta_var,
        original_depths_shape =  original_depths_shape,
        original_poses_shape = original_poses_shape
    )

    #print("HELLOOOO",cost_fn.optim_vars)
    #print(type(cost_fn))
    # Objective
    objective = th.Objective().to("cpu")

    
    objective.add(cost_function=cost_fn)
    objective.update({"poses":poses_var.tensor,"depths":disps_var.tensor})
    #print("BATCHHHHHH :",objective.batch_size)
    # Optimizer
    optimizer = th.LevenbergMarquardt(
    objective.to(torch.float64),
        max_iterations=10,
        step_size=1,
        vectorize=True)
    # Optimize
    optimizer.optimize()

    # Extract results
    optimized_poses = poses_var.tensor.view_as(poses)
    optimized_disps = disps_var.tensor.view_as(disps)

    return optimized_poses, optimized_disps







# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


def BA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Full Bundle Adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)
    
    r = (target - coords).reshape(B, N, -1,1)
    w = .001 * (valid * weight).reshape(B, N, -1,1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    w = w.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(w*r*Jz, dim=-1)
    Ck = torch.sum(w*Jz*Jz, dim=-1)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    C = C + eta.view(*C.shape) + 1e-7

    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht*wd)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w)
    
    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps


def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses




def levenberg_marquardt_step(poses, disps, target, ii, jj, intrinsics,weight, eta, lambda_init=1e-5, fixedp=1, rig=1,max_ittr=1):
    """
    Perform a Levenberg-Marquardt step for dense bundle adjustment.
    """    
    # Initialize damping factor
    lambda_ = lambda_init
    max_iter = max_ittr
    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)
    
    r = (target - coords).reshape(B, N, -1,1)
    w = .001 * (valid * weight).reshape(B, N, -1,1)
    prev_cost =torch.sum(torch.abs(r))


    for _ in range(max_iter):
        
        B, P, ht, wd = disps.shape
        N = ii.shape[0]
        D = poses.manifold_dim

        ### 1: commpute jacobians and residuals ###
        coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
            poses, disps, intrinsics, ii, jj, jacobian=True)
        r = (target - coords).reshape(B, N, -1,1)
        w = .001 * (valid * weight).reshape(B, N, -1,1)

        ### 2: construct linear system ###
        Ji = Ji.reshape(B, N, -1, D)
        Jj = Jj.reshape(B, N, -1, D)
        wJiT = (w * Ji).transpose(2,3)
        wJjT = (w * Jj).transpose(2,3)

        Jz = Jz.reshape(B, N, ht*wd, -1)

        Hii = torch.matmul(wJiT, Ji)
        Hij = torch.matmul(wJiT, Jj)
        Hji = torch.matmul(wJjT, Ji)
        Hjj = torch.matmul(wJjT, Jj)

        vi = torch.matmul(wJiT, r).squeeze(-1)
        vj = torch.matmul(wJjT, r).squeeze(-1)

        Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
        Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

        w = w.view(B, N, ht*wd, -1)
        r = r.view(B, N, ht*wd, -1)
        wk = torch.sum(w*r*Jz, dim=-1)
        Ck = torch.sum(w*Jz*Jz, dim=-1)

        kx, kk = torch.unique(ii, return_inverse=True)
        M = kx.shape[0]

        # only optimize keyframe poses
        P = P // rig - fixedp
        ii = ii // rig - fixedp
        jj = jj // rig - fixedp

        H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
            safe_scatter_add_mat(Hij, ii, jj, P, P) + \
            safe_scatter_add_mat(Hji, jj, ii, P, P) + \
            safe_scatter_add_mat(Hjj, jj, jj, P, P)

        E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
            safe_scatter_add_mat(Ej, jj, kk, P, M)

        v = safe_scatter_add_vec(vi, ii, P) + \
            safe_scatter_add_vec(vj, jj, P)

        C = safe_scatter_add_vec(Ck, kk, M)
        w = safe_scatter_add_vec(wk, kk, M)

        C = C + eta.view(*C.shape) + 1e-7

        H = H.view(B, P, P, D, D)
        H_damped = H + lambda_ * torch.eye(H.shape[0], device=H.device)
        E = E.view(B, P, M, D, ht*wd)

        ### 3: solve the system ###
        dx, dz = schur_solve(H_damped, E, C, v, w)
        
        ### 4: apply retraction ###
        poses[...] = pose_retr(poses, dx, torch.arange(P) + fixedp)
        disps[...] = disp_retr(disps, dz.view(B,-1,ht,wd), kx)


        disps[...] = torch.where(disps > 10, torch.zeros_like(disps), disps)
        disps[...] = disps.clamp(min=0.0)

        coords, valid, (Ji, Jj, Jz) = pops.projective_transform(poses, disps, intrinsics, ii, jj, jacobian=True)
        
        new_r=(target - coords).reshape(B, N, -1,1)
        
        new_cost = torch.sum(torch.abs(new_r))
        
        
        # If the new cost is smaller, accept the step and update parameters
        if new_cost < prev_cost:
            poses[...] = poses
            disps[...] = disps
            prev_cost = new_cost
            lambda_ = lambda_ / 10  # Decrease damping factor for larger steps
        else:
            lambda_ = lambda_ * 10  # Increase damping factor for smaller steps

    return poses, disps


