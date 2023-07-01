
# https://github.com/google/hypernerf/blob/59ca296c6cf4369f81177b928656aea62d085fd2/hypernerf/rigid_body.py

import torch
import numpy as np
from einops import rearrange, reduce, repeat 


def skew(w):
    """
    Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.
    Args:
    w: (3,) A 3-vector
    Returns:
    W: (3, 3) A skew matrix such that W @ v == w x v

    """
    # w = jnp.reshape(w, (3))

    w = w.reshape(3)

    W = torch.tensor([[0.0, -w[2], w[1]], \
                   [w[2], 0.0, -w[0]], \
                   [-w[1], w[0], 0.0]],
                   dtype=w.dtype,device=w.device) 
    return W 


def exp_so3(w, theta):
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.
    Args:
    w: (3,) An axis of rotation. This is assumed to be a unit-vector.
    theta: An angle of rotation. (float)
    Returns:
    R: (3, 3) An orthonormal rotation matrix representing a rotation of
      magnitude theta about axis w.
    """

    # 3,3
    W = skew(w)

    W_sq = torch.matmul(W,W)

    # return (
    #   jnp.eye(3)
    #   + jnp.sin(theta) * W 
    #   + (1.0 - jnp.cos(theta)) * matmul(W, W))

    mat = ( torch.eye(3,device=w.device,dtype=w.dtype) + 
            torch.sin(theta) * W + 
            (1.0 - torch.cos(theta))*W_sq 
           )

    return mat 


def rp_to_se3(R, p) :
  """Rotation and translation to homogeneous transform.
  Args:
    R: (3, 3) An orthonormal rotation matrix.
    p: (3,) A 3-vector representing an offset.
  Returns:
    X: (4, 4) The homogeneous transformation matrix described by rotating by R and translating by p.
  """
  # p = jnp.reshape(p, (3, 1))
  p = p.reshape(3,1)

  X = torch.eye(4,device=p.device,dtype=p.dtype)

  X[:3,:3]=R
  X[:3, 3]=p
  # return jnp.block([[R, p], [jnp.array([[0.0, 0.0, 0.0, 1.0]])]])
  return X 


def exp_se3( w, v, theta ):
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.88.
    Args:
    S: (6,) A screw axis of motion.
    theta: Magnitude of motion. (float)
    Returns:
    a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating motion of magnitude theta about S for one second.
    """
    # w, v = jnp.split(S, 2)

    device = w.device

    W = skew(w)
    W_sq = torch.matmul(W,W) 

    R = exp_so3(w, theta) 

    # p = matmul( 
    #       (
    #           theta * jnp.eye(3) 
    #           + (1.0 - jnp.cos(theta)) * W 
    #           + (theta - jnp.sin(theta)) * matmul(W, W)
    #       ), v)
    #           

    eye3 = torch.eye(3,device=w.device,dtype=w.dtype)

    a = ( theta*eye3 + 
        (1.0- torch.cos(theta))*W + 
        (theta - torch.sin(theta))*W_sq )

    v = v.reshape(3,1)
    p = torch.matmul(a,v) 

    #return rp_to_se3(R, p)
    return R,p



def b_skew(w):
    """
    Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.
    Args:
    w: (3,) A 3-vector
    Returns:
    W: (3, 3) A skew matrix such that W @ v == w x v

    """
    # w = jnp.reshape(w, (3))

    assert w.ndim==2 and w.shape[1]==3

    W = torch.zeros( (w.shape[0],3,3),
                       dtype=w.dtype, 
                       device=w.device) 
    
    W[:,0,1]=-1*w[:,2]
    W[:,0,2]=   w[:,1]

    W[:,1,0]=   w[:,2]
    W[:,1,2]=-1*w[:,0]
    
    W[:,2,0]=-1*w[:,1]
    W[:,2,1]=   w[:,0]

    # W = torch.tensor([[0.0, -w[2], w[1]], \
    #                   [w[2], 0.0, -w[0]], \
    #                   [-w[1], w[0], 0.0] ],
    #                dtype=w.dtype,device=w.device) 
    return W 

def b_exp_so3(w, theta):
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.
    Args:
    w: (b,3) An axis of rotation. This is assumed to be a unit-vector.
    theta: An angle of rotation. (b,)
    
    Returns:
    R: (b, 3, 3) An orthonormal rotation matrix representing a rotation of magnitude theta about axis w.
    """

    B = w.shape[0]
    assert theta.ndim ==1 and theta.shape[0] == B


    # B,3,3
    W = b_skew(w)

    W_sq = torch.bmm(W,W)

    # return (
    #   jnp.eye(3)
    #   + jnp.sin(theta) * W 
    #   + (1.0 - jnp.cos(theta)) * matmul(W, W))
    eye3  = torch.eye(3,device=w.device,dtype=w.dtype)
    beye3 = repeat(eye3, 'r c -> b r c ', b=B)

    # mat = ( eye3 + 
    #         torch.sin(theta) * W + 
    #         (1.0 - torch.cos(theta))*W_sq 
    #        )
    sin_theta = torch.sin(theta)
    sin_theta = rearrange(sin_theta,'b -> b 1 1')

    cos_theta = torch.cos(theta)
    cos_theta = rearrange(cos_theta,'b -> b 1 1')

    mat = ( beye3 + 
            sin_theta * W + 
            (1.0 - cos_theta)*W_sq 
           )

    return mat 



def b_exp_se3( w, v, theta ):
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.88.
    Args:
    S:  A screw axis of motion.
    w: (b,3)
    v: (b,3)
    theta: Magnitude of motion. (b, )

    Returns:
    a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating motion of magnitude theta about S for one second.
    R: b,3,3
    v: b,3.1

    """
    # w, v = jnp.split(S, 2)

    B = w.shape[0]
    
    assert theta.ndim ==1 and theta.shape[0] == B

    #----------------------------------

    device = w.device

    W = b_skew(w)
    W_sq = torch.bmm(W,W) 

    # b,3,3
    R = b_exp_so3(w, theta) 

    #---------------------------------- 
    eye3  = torch.eye(3,device=w.device,dtype=w.dtype)
    beye3 = repeat(eye3, 'r c -> b r c ', b=B)

    theta2    = rearrange(theta, 'b -> b 1 1')

    sin_theta = torch.sin(theta2)  
    cos_theta = torch.cos(theta2) 

    a = ( theta2*beye3 + 
        ( 1.0    - cos_theta)*W + 
        ( theta2 - sin_theta)*W_sq )

    v = rearrange(v,'b c -> b c 1')

    # B,3,1
    p = torch.bmm(a,v) 

    #----------------------------------
    # p = matmul( 
    #       (
    #           theta * jnp.eye(3) 
    #           + (1.0 - jnp.cos(theta)) * W 
    #           + (theta - jnp.sin(theta)) * matmul(W, W)
    #       ), v)
    #           

    return R,p
