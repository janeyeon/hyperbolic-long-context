#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from loguru import logger


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture_curv(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.

    # min_radius = torch.clamp(min_radius, min=1e-8)
    #! 상수로 고정! 
    min_radius = 0.1
    # min_radius = 0.5

    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture

def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.

    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    return _angle

import torch
from torch import Tensor

def xoy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `y` in the hyperbolic triangle `xOy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Time components (0-th dim in hyperboloid embedding) scaled by curvature
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Lorentzian inner product scaled by curvature
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Numerator and denominator for acos argument (hyperbolic law of cosines)
    acos_numer = x_time + c_xyl * y_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    # Final input to arccos
    acos_input = acos_numer / (torch.norm(y, dim=-1) * acos_denom + eps)

    # Clamp and get angle
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    return _angle


# def oxy_angle(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor = 1.0, eps: float = 1e-8):
#     x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
#     y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

#     print("x_time:", x_time.min().item(), x_time.max().item(), "NaN?", torch.isnan(x_time).any().item())
#     print("y_time:", y_time.min().item(), y_time.max().item(), "NaN?", torch.isnan(y_time).any().item())

#     c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
#     print("c_xyl:", c_xyl.min().item(), c_xyl.max().item(), "NaN?", torch.isnan(c_xyl).any().item())

#     acos_numer = y_time + c_xyl * x_time
#     print("acos_numer:", acos_numer.min().item(), acos_numer.max().item(), "NaN?", torch.isnan(acos_numer).any().item())

#     acos_denom_raw = c_xyl**2 - 1
#     acos_denom = torch.sqrt(torch.clamp(acos_denom_raw, min=eps))
#     print("acos_denom_raw:", acos_denom_raw.min().item(), acos_denom_raw.max().item(), "NaN?", torch.isnan(acos_denom_raw).any().item())
#     print("acos_denom:", acos_denom.min().item(), acos_denom.max().item(), "NaN?", torch.isnan(acos_denom).any().item())

#     norm_x = torch.norm(x, dim=-1)
#     print("norm_x:", norm_x.min().item(), norm_x.max().item(), "NaN?", torch.isnan(norm_x).any().item())

#     acos_input = acos_numer / (norm_x * acos_denom + eps)
#     print("acos_input (before clamp):", acos_input.min().item(), acos_input.max().item(), "NaN?", torch.isnan(acos_input).any().item())

#     acos_input = torch.clamp(acos_input, min=-1 + eps, max=1 - eps)
#     _angle = torch.acos(acos_input)

#     print("angle:", _angle.min().item(), _angle.max().item(), "NaN?", torch.isnan(_angle).any().item())

#     return _angle

def oxy_angle_eval(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))

    logger.info(f"x_time shape: {x_time.size()}")
    logger.info(f"y_time shape: {y_time.size()}")

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).

    # c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    c_xyl = curv * (y @ x.T - y_time @ x_time.T)
    logger.info(f"c_xyl shape: {c_xyl.size()}")

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time.T
    logger.info(f"acos_numer shape: {acos_numer.size()}")
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    logger.info(f"acos_denom shape: {acos_denom.size()}")

    acos_input = acos_numer / (torch.norm(x, dim=-1, keepdim=True).T * acos_denom + eps)
    _angle = - torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle

def pairwise_oxy_angle_matrix(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, eps: float = 1e-8):
    """
    ϕ(x,y): x에서 본 외접각을 배치쌍으로 계산. shape (B1,B2)
    """
    x_time = torch.sqrt(1 / curv + (x**2).sum(-1, keepdim=True))   # (B1,1)
    y_time = torch.sqrt(1 / curv + (y**2).sum(-1, keepdim=True))   # (B2,1)

    c = curv * (x @ y.T - x_time @ y_time.T)                       # (B1,B2)
    acos_numer = y_time.T + c * x_time     # (B1,B2)
    acos_denom = torch.sqrt(torch.clamp(c**2 - 1, min=eps))        # (B1,B2)
    norm_x = x.norm(dim=-1, keepdim=True)                          # (B1,1)

    acos_input = acos_numer / (norm_x * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1+eps, max=1-eps))  # (B1,B2)

def symmetric_oxy_angle(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, eps: float = 1e-8):
    """
    ϕ_sym(x,y) = 0.5*(ϕ(x,y) + ϕ(y,x))  (대칭화)
    """
    phi_xy = pairwise_oxy_angle_matrix(x, y, curv, eps)
    phi_yx = pairwise_oxy_angle_matrix(y, x, curv, eps).T
    return 0.5 * (phi_xy + phi_yx)



def symmetric_oxy_angle_5(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, eps: float = 1e-8):
    """
    ϕ_sym(x,y) = 0.5*(ϕ(x,y) + ϕ(y,x))  (대칭화)
    """
    phi_xy = pairwise_oxy_angle_matrix(x, y, curv, eps)
    # phi_yx = pairwise_oxy_angle_matrix(y, x, curv, eps).T
    # return 0.5 * (phi_xy + phi_yx)

    phi_sum = torch.sin(phi_xy) 
    return phi_sum




def symmetric_oxy_angle_6(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, eps: float = 1e-8):
    """
    ϕ_sym(x,y) = 0.5*(ϕ(x,y) + ϕ(y,x))  (대칭화)
    """
    phi_xy = pairwise_oxy_angle_matrix(x, y, curv, eps)
    phi_yx = pairwise_oxy_angle_matrix(y, x, curv, eps).T
    # return 0.5 * (phi_xy + phi_yx)

    phi_sum = 0.5 * (torch.sin(phi_xy) + torch.sin(phi_yx))
    return phi_sum



def symmetric_oxy_angle_7(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, eps: float = 1e-8):
    """
    ϕ_sym(x,y) = 0.5*(ϕ(x,y) + ϕ(y,x))  (대칭화)
    """
    phi_xy = pairwise_oxy_angle_matrix(x, y, curv, eps)
    phi_yx = pairwise_oxy_angle_matrix(y, x, curv, eps).T
    # return 0.5 * (phi_xy + phi_yx)

    diff_angle = torch.pi - (phi_xy + phi_yx)

    phi_sum = 0.5 * (torch.sin(phi_xy) + torch.sin(phi_yx))
    return phi_sum




def symmetric_oxy_angle_9(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, eps: float = 1e-8):
    """
    ϕ_sym(x,y) = 0.5*(ϕ(x,y) + ϕ(y,x))  (대칭화)
    """
    phi_xy = pairwise_oxy_angle_matrix(x, y, curv, eps)
    # return 0.5 * (phi_xy + phi_yx)


    phi_sum = (torch.cos(phi_xy * 2))
    return phi_sum



def symmetric_oxy_angle_10(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, eps: float = 1e-8):
    """
    ϕ_sym(x,y) = 0.5*(ϕ(x,y) + ϕ(y,x))  (대칭화)
    """
    phi_xy = pairwise_oxy_angle_matrix(x, y, curv, eps)
    # return 0.5 * (phi_xy + phi_yx)


    phi_sum = (-torch.cos(phi_xy))
    return phi_sum

def joint_logits_dist_angle(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, lam_angle: float = 0.5, eps: float = 1e-8):
    """
    s(x,y) = -( sqrt(curv)*d(x,y) + lam_angle * ϕ_sym(x,y) )
    반환: 로짓(큰 값일수록 유사)
    """
    d = pairwise_dist(x, y, curv, eps)                  # (B1,B2)
    # d_scaled = (curv**0.5) * d                          # 무차원화
    theta = symmetric_oxy_angle(x, y, curv, eps)        # (B1,B2)
    
    # print(f"d: {d.mean()}, lam_angle: {lam_angle}, theta: {theta.mean()}")

    return -(d + (lam_angle**2) * theta)

def angle_only_logits(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor, symmetric: bool = True, eps: float = 1e-8):
    """
    s(x,y) = -ϕ(x,y) 또는 -ϕ_sym(x,y)
    """
    if symmetric:
        theta = symmetric_oxy_angle(x, y, curv, eps)
    else:
        theta = pairwise_oxy_angle_matrix(x, y, curv, eps)
    return -theta



def pairwise_angle_at_origin(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    """
    Δθ_O(x,y): 원점 O에서 본 각 차이. x,y는 '공간 성분'(네 코드와 동일 가정).
    cos Δθ = <x,y>/ (||x|| ||y||)
    반환 shape: (B1,B2)
    """
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)      # (B1,1)
    y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)      # (B2,1)
    cos_th = (x @ y.T) / (x_norm * y_norm.T)                  # (B1,B2)
    cos_th = torch.clamp(cos_th, -1+eps, 1-eps)
    return torch.acos(cos_th)                                  # (B1,B2)

def radial_to_origin(x: torch.Tensor, curv: float | torch.Tensor, eps: float = 1e-8):
    """
    r(x) = dist(O,x). x는 공간성분, x_time은 네 코드와 동일 계산 사용.
    반환 shape: (B,1)
    """
    x_time = torch.sqrt(1 / curv + (x**2).sum(-1, keepdim=True))   # (B,1)
    z = (curv**0.5) * x_time
    z = torch.clamp(z, min=1+eps)                                  # acosh domain
    return torch.acosh(z) / (curv**0.5)


def polar_distance_law_of_cosines(
    x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor,
    theta_mode: str = "origin",   # "origin" | "oxy_sym"
    eps: float = 1e-8
):
    """
    d_polar_LOS(x,y): cosh(√κ d) = cosh(√κ r_x)cosh(√κ r_y) - sinh(√κ r_x)sinh(√κ r_y) cos(Δθ)
    theta_mode:
      - "origin": Δθ = 원점 O에서 본 각 (권장, 정확한 하이퍼볼릭 법코사인)
      - "oxy_sym": Δθ = ϕ_sym(x,y) (네가 정의한 Oxy 외접각의 대칭화로 대체하고 싶을 때)
    반환 shape: (B1,B2)
    """
    r_x = radial_to_origin(x, curv, eps)           # (B1,1)
    r_y = radial_to_origin(y, curv, eps)           # (B2,1)

    if theta_mode == "origin":
        theta = pairwise_angle_at_origin(x, y, eps)        # (B1,B2)
    elif theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle(x, y, curv, eps)       # (B1,B2)
    else:
        raise ValueError("theta_mode must be 'origin' or 'oxy_sym'.")

    rx = (curv**0.5) * r_x                               # (B1,1)
    ry = (curv**0.5) * r_y                               # (B2,1)
    cosh_rx = torch.cosh(rx)                             # (B1,1)
    cosh_ry = torch.cosh(ry)                             # (B2,1)
    sinh_rx = torch.sinh(rx)                             # (B1,1)
    sinh_ry = torch.sinh(ry)                             # (B2,1)

    cos_term = torch.cos(theta)                          # (B1,B2)

    inside = cosh_rx * cosh_ry.T - (sinh_rx * sinh_ry.T) * cos_term  # (B1,B2)
    inside = torch.clamp(inside, min=1+eps)              # acosh domain 안정화
    return torch.acosh(inside) / (curv**0.5)             # (B1,B2)

def polar_distance_quadratic(
    x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor,
    # theta_mode: str = "origin",   # "origin" | "oxy_sym"
    theta_mode: str = "oxy_sym",   # "origin" | "oxy_sym"

    eps: float = 1e-8
):
    """
    d_polar_Q(x,y) ≈ sqrt( (r_x - r_y)^2 + [S_kappa(r̄)]^2 * (Δθ)^2 ),  r̄=(r_x+r_y)/2
    S_kappa(r) = sinh(√κ r)/√κ
    반환 shape: (B1,B2)
    """
    r_x = radial_to_origin(x, curv, eps)     # (B1,1)
    r_y = radial_to_origin(y, curv, eps)     # (B2,1)

    if theta_mode == "origin":
        theta = pairwise_angle_at_origin(x, y, eps)      # (B1,B2)
    elif theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle(x, y, curv, eps)     # (B1,B2)
    else:
        raise ValueError("theta_mode must be 'origin' or 'oxy_sym'.")

    r_bar = 0.5 * (r_x + r_y.T)                          # (B1,B2)
    S = torch.sinh((curv**0.5) * r_bar) / (curv**0.5)    # (B1,B2)

    rho = torch.abs(r_x - r_y.T)                         # (B1,B2)
    return torch.sqrt(rho**2 + (S**2) * (theta**2) + eps)



def angular_weight(
    theta: torch.Tensor,
    mode: str = "softcone",      # "softcone" | "gauss"
    sigma: float = 1.0,          # gauss 폭(라디안)
    omega: float | None = None,  # softcone 반각(라디안). None이면 sigma로 유도
    k: float = 8.0,              # softcone 경사(클수록 경계가 뚜렷)
    eps: float = 1e-8,
):
    """
    θ -> W(θ) in (0,1].  θ 작을수록 W 큼.
    - softcone: W = sigmoid(k*(ω - θ))  (콘 밖에서도 부드럽게 감소, 0은 아님)
    - gauss:    W = exp(-(θ/σ)^2)       (원형으로 부드럽게 감소)
    """
    if mode == "softcone":
        if omega is None:
            # sigma(~표적 반각)로부터 ω 유도: θ=σ에서 W≈0.5 되도록
            omega = sigma
        return torch.sigmoid(k * (omega - theta))
    elif mode == "gauss":
        return torch.exp(- (theta / (sigma + eps))**2)
    else:
        raise ValueError("mode must be 'softcone' or 'gauss'")
    

def dist_angle_product_logits(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    # oxy_sym : 포함/엔테일먼트(콘) 해석을 강조, 반지름 차이가 크고, 로컬 방향 정렬이 중요한 데이터
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    # "softcone": 콘 안팎을 명확히 구분하면서도 바깥에서도 미분 신호 유지, 엔테일먼트/부분순서처럼 “방향 제약”을 강하게 주고 싶을 때, negative가 콘 바깥으로 널리 퍼지는 과제
    # "gauss": 노이즈에 강함, 어디서나 그라디언트 매끄러움, 라벨/정렬 노이즈가 큰 데이터, 초기 학습 안정화(커리큘럼 시작점)
    ang_sigma: float = 0.8,      # 각도 폭(라디안). softcone일 땐 반각 역할
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치(지수)
    center: bool = False,        # 로짓 안정화용 중심화
    eps: float = 1e-8,
):
    """
    sim(x,y) = exp(-√κ d / τ_d) * [ W(θ) ]^λ
    logits   = log sim = -(√κ d)/τ_d + λ * log W(θ)
    -> 가까울수록↑, θ 작을수록↑, 콘 밖에서도 연속적으로↓ (W>0)
    """
    # 하이퍼볼릭 거리 (B1,B2)
    d = pairwise_dist(x, y, curv, eps)               # (단위: 1/√κ)
    d_dimless = (curv**0.5) * d                      # 무차원화

    # 각도 선택
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle(x, y, curv, eps)   # (B1,B2)
    elif theta_mode == "origin":
        # 원점 기준 방향 차이(필요시)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1+eps, 1-eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    # 각도 가중치 (0,1]
    W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma, omega=ang_sigma, k=ang_k, eps=eps)
    W = torch.clamp(W, min=eps, max=1.0)

    logits = - d_dimless / (tau_d + eps) + lam * torch.log(W) * d_dimless.mean()
    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W)

    if center:
        # 행/열 평균 제거로 안정화(선택)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    lam_angle: float = 1.0,
    eps: float = 1e-8,
    ang_sigma=0.8,
    ang_k=8.0,
):
    # 기존 시그니처 유지. 내부는 product-of-experts 로짓 사용
    return dist_angle_product_logits(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        # ang_mode="softcone",
        ang_mode="gauss",
        ang_sigma=ang_sigma,
        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        center=True,
        eps=eps,
    )

def compute_mahalanobis_inverse(tangent_vectors: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Compute inverse covariance matrix (Σ⁻¹) from tangent space vectors.
    Args:
        tangent_vectors: Tensor of shape (B, D)
    Returns:
        Inverse covariance matrix (D, D)
    """
    B, D = tangent_vectors.shape
    mean = tangent_vectors.mean(dim=0, keepdim=True)
    centered = tangent_vectors - mean  # (B, D)
    cov = centered.T @ centered / (B - 1)  # (D, D)
    cov += eps * torch.eye(D, device=tangent_vectors.device)  # regularization
    return torch.inverse(cov)

def hmpc_pairwise_distance(
    x: Tensor,  # (B1, D)
    y: Tensor,  # (B2, D)
    curv: float = 1.0,
    eps: float = 1e-8
) -> Tensor:
    """
    Compute pairwise hyperbolic Mahalanobis polar coordinate distance matrix.
    Returns:
        distance: (B1, B2)
    """
    B1, D = x.shape
    B2, _ = y.shape

    # 1. Project to tangent space
    v_x = log_map0(x, curv)  # (B1, D)
    v_y = log_map0(y, curv)  # (B2, D)

    # 2. Compute joint inverse covariance from all tangent vectors
    v_all = torch.cat([v_x, v_y], dim=0)  # (B1 + B2, D)
    Sigma_inv = compute_mahalanobis_inverse(v_all)  # (D, D)

    # 3. Mahalanobis norm
    r_x = torch.sqrt(torch.sum(v_x @ Sigma_inv * v_x, dim=-1, keepdim=True) + eps)  # (B1, 1)
    r_y = torch.sqrt(torch.sum(v_y @ Sigma_inv * v_y, dim=-1, keepdim=True) + eps)  # (B2, 1)

    # # 4. Inner product (B1, B2)
    inner = v_x @ Sigma_inv @ v_y.T  # (B1, B2)
    denom = r_x @ r_y.T + eps  # (B1, B2)
    cos_theta = torch.clamp(inner / denom, min=-1 + eps, max=1 - eps)
    # cos_theta = torch.cos(symmetric_oxy_angle(x, y, curv, eps))

    # 5. Final Mahalanobis polar distance
    dist = torch.sqrt(r_x**2 + r_y.T**2 - 2 * r_x * r_y.T * cos_theta + eps)  # (B1, B2)

    return dist


def dist_angle_product_logits_margin(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    ang_sigma: float = 0.8,      # 각도 폭
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치
    margin: float = 0.2,         # 기본 마진
    margin_mode: str = "geodesic",  # "geodesic" | "radius"
    alpha: float = 0.5,          # 반경 기반 margin scaling
    center: bool = False,
    eps: float = 1e-8,
):
    """
    Hyperbolic distance + angular weighting + geodesic margin.
    sim(x,y) = exp(-(√κ * (d - m)) / τ_d) * [W(θ)]^λ
    logits   = -(√κ * max(0, d - m)) / τ_d + λ * log W(θ)
    """
    # -----------------------------
    # 1. Geodesic distance
    # -----------------------------
    d = pairwise_dist(x, y, curv, eps)          # (B1, B2)
    d_dimless = (curv**0.5) * d                 # 무차원화

    # -----------------------------
    # 2. Angle weighting
    # -----------------------------
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle(x, y, curv, eps)
    elif theta_mode == "origin":
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1 + eps, 1 - eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma,
                       omega=ang_sigma, k=ang_k, eps=eps)
    W = torch.clamp(W, min=eps, max=1.0)

    # -----------------------------
    # 3. Margin computation
    # -----------------------------
    if margin_mode == "geodesic":
        # 지오데식 거리 기반 margin
        m = margin
    elif margin_mode == "radius":
        # 반경 기반 margin: 반경이 클수록 더 큰 margin 적용
        r_x = radial_to_origin(x, curv, eps)    # (B1, 1)
        r_y = radial_to_origin(y, curv, eps)    # (B2, 1)
        r_mean = 0.5 * (r_x + r_y.T)            # (B1, B2)
        m = margin * torch.tanh(alpha * r_mean)
    else:
        raise ValueError("margin_mode must be 'geodesic' or 'radius'.")

    # -----------------------------
    # 4. Margin-aware logits
    # -----------------------------
    d_margin = torch.clamp(d_dimless - m, min=0.0)
    logits = - d_margin / (tau_d + eps) + lam * torch.log(W)

    # -----------------------------
    # 5. Centering (optional)
    # -----------------------------
    if center:
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle_margin(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    ang_sigma: torch.Tensor,
    ang_k: torch.Tensor,
    lam_angle: float = 1.0,
    margin: float = 0.2,
    eps: float = 1e-8
):
    return dist_angle_product_logits_margin(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        ang_mode="gauss",
        ang_sigma=ang_sigma,
        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        margin=margin,
        margin_mode="radius",   # 반경 기반 margin 적용
        alpha=0.5,
        center=True,
        eps=eps,
    )


def dist_angle_product_logits_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    # oxy_sym : 포함/엔테일먼트(콘) 해석을 강조, 반지름 차이가 크고, 로컬 방향 정렬이 중요한 데이터
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    # "softcone": 콘 안팎을 명확히 구분하면서도 바깥에서도 미분 신호 유지, 엔테일먼트/부분순서처럼 “방향 제약”을 강하게 주고 싶을 때, negative가 콘 바깥으로 널리 퍼지는 과제
    # "gauss": 노이즈에 강함, 어디서나 그라디언트 매끄러움, 라벨/정렬 노이즈가 큰 데이터, 초기 학습 안정화(커리큘럼 시작점)
    ang_sigma: float = 0.8,      # 각도 폭(라디안). softcone일 땐 반각 역할
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치(지수)
    center: bool = False,        # 로짓 안정화용 중심화
    eps: float = 1e-8,
):
    """
    sim(x,y) = exp(-√κ d / τ_d) * [ W(θ) ]^λ
    logits   = log sim = -(√κ d)/τ_d + λ * log W(θ)
    -> 가까울수록↑, θ 작을수록↑, 콘 밖에서도 연속적으로↓ (W>0)
    """
    # 하이퍼볼릭 거리 (B1,B2)
    d = pairwise_dist(x, y, curv, eps)               # (단위: 1/√κ)
    d_dimless = (curv**0.5) * d                      # 무차원화

    # 각도 선택
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle(x, y, curv, eps)   # (B1,B2)
    elif theta_mode == "origin":
        # 원점 기준 방향 차이(필요시)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1+eps, 1-eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    # 각도 가중치 (0,1]
    W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma, omega=ang_sigma, k=ang_k, eps=eps)
    W = torch.clamp(W, min=eps, max=1.0)

    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W) * d_dimless.mean()
    # def fused_effective_distance(d, W, lambda_w=0.5, beta=0.3, eps=1e-8):
    #     # W in [0,1]
    #     W = W.clamp_min(eps).clamp_max(1.0)
    #     # S(W,λ) = 1 - β * [1 - (1 - W)^λ]
    #     # S = 1.0 - beta * (1.0 - torch.pow(1.0 - W, lambda_w))
    #     S = 1.0 - d.mean() * (1.0 - torch.pow(1.0 - W, lambda_w))
    #     return d * S
    # effective_distance = d * (1 - W ** lam)
    # effective_distance = fused_effective_distance(d, W, lam)

    
    # logits = - torch.sqrt(curv) * effective_distance / (tau_d + eps)

    logits = - (d_dimless * (1 - lam * (1 - W))) / (tau_d + eps)

    if center:
        # 행/열 평균 제거로 안정화(선택)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    ang_sigma: torch.Tensor,
    ang_k: torch.Tensor,
    lam_angle: float = 1.0,
    eps: float = 1e-8
):
    # 기존 시그니처 유지. 내부는 product-of-experts 로짓 사용
    return dist_angle_product_logits_mul(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        # ang_mode="softcone",
        ang_mode="gauss",
        ang_sigma=ang_sigma,
        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        center=True,
        eps=eps,
    )


def dist_angle_product_logits_mean(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    # oxy_sym : 포함/엔테일먼트(콘) 해석을 강조, 반지름 차이가 크고, 로컬 방향 정렬이 중요한 데이터
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    # "softcone": 콘 안팎을 명확히 구분하면서도 바깥에서도 미분 신호 유지, 엔테일먼트/부분순서처럼 “방향 제약”을 강하게 주고 싶을 때, negative가 콘 바깥으로 널리 퍼지는 과제
    # "gauss": 노이즈에 강함, 어디서나 그라디언트 매끄러움, 라벨/정렬 노이즈가 큰 데이터, 초기 학습 안정화(커리큘럼 시작점)
    ang_sigma: float = 0.8,      # 각도 폭(라디안). softcone일 땐 반각 역할
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치(지수)
    center: bool = False,        # 로짓 안정화용 중심화
    eps: float = 1e-8,
):
    """
    sim(x,y) = exp(-√κ d / τ_d) * [ W(θ) ]^λ
    logits   = log sim = -(√κ d)/τ_d + λ * log W(θ)
    -> 가까울수록↑, θ 작을수록↑, 콘 밖에서도 연속적으로↓ (W>0)
    """
    # 하이퍼볼릭 거리 (B1,B2)
    d = pairwise_dist(x, y, curv, eps)               # (단위: 1/√κ)
    d_dimless = (curv**0.5) * d                      # 무차원화

    # 각도 선택
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle(x, y, curv, eps)   # (B1,B2)
    elif theta_mode == "origin":
        # 원점 기준 방향 차이(필요시)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1+eps, 1-eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    x_norm = x.norm().clamp_min(eps)
    y_norm = y.norm().clamp_min(eps)
    lam = (x_norm + y_norm) / 30
    # lam = (x_norm + y_norm) / 50

    # 각도 가중치 (0,1]
    print(f"theta: {theta}")
    print(f"lam: {lam}")
    
    omega = 2.0
    W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma, omega=omega, k=ang_k, eps=eps)
    # W = torch.clamp(W, min=eps, max=1.0)
    # print(f"W before: {W}")
    W = (W - W.min()) / (W.max() - W.min())
    # print(f"W after: {W}")

    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W) * d_dimless.mean()

    logits = - d_dimless * (1 + lam * W) / (tau_d + eps) 

    # print(f"d_dimless: {d_dimless}")
    # print(f"tau_d: {tau_d}")
    # print(f"logits: {logits}")



    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W)

    if center:
        # 행/열 평균 제거로 안정화(선택)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle_mean(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    lam_angle: float = 1.0,
    eps: float = 1e-8,
    ang_sigma=0.8,
    ang_k=8.0,
):
    # 기존 시그니처 유지. 내부는 product-of-experts 로짓 사용
    return dist_angle_product_logits_mean(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        # ang_mode="softcone",
        ang_mode="gauss",
        # ang_sigma=ang_sigma,
        ang_sigma=2.0,

        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        center=True,
        eps=eps,
    )

def dist_angle_product_logits_mean_5(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    # oxy_sym : 포함/엔테일먼트(콘) 해석을 강조, 반지름 차이가 크고, 로컬 방향 정렬이 중요한 데이터
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    # "softcone": 콘 안팎을 명확히 구분하면서도 바깥에서도 미분 신호 유지, 엔테일먼트/부분순서처럼 “방향 제약”을 강하게 주고 싶을 때, negative가 콘 바깥으로 널리 퍼지는 과제
    # "gauss": 노이즈에 강함, 어디서나 그라디언트 매끄러움, 라벨/정렬 노이즈가 큰 데이터, 초기 학습 안정화(커리큘럼 시작점)
    ang_sigma: float = 0.8,      # 각도 폭(라디안). softcone일 땐 반각 역할
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치(지수)
    center: bool = False,        # 로짓 안정화용 중심화
    eps: float = 1e-8,
):
    """
    sim(x,y) = exp(-√κ d / τ_d) * [ W(θ) ]^λ
    logits   = log sim = -(√κ d)/τ_d + λ * log W(θ)
    -> 가까울수록↑, θ 작을수록↑, 콘 밖에서도 연속적으로↓ (W>0)
    """
    # 하이퍼볼릭 거리 (B1,B2)
    d = pairwise_dist(x, y, curv, eps)               # (단위: 1/√κ)
    d_dimless = (curv**0.5) * d                      # 무차원화

    # 각도 선택
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle_5(x, y, curv, eps)   # (B1,B2)
    elif theta_mode == "origin":
        # 원점 기준 방향 차이(필요시)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1+eps, 1-eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    x_norm = x.norm().clamp_min(eps)
    y_norm = y.norm().clamp_min(eps)
    # lam = (x_norm + y_norm) / 30
    lam = (x_norm + y_norm) / 30

    # 각도 가중치 (0,1]
    # print(f"theta: {theta}")
    # print(f"lam: {lam}")
    
    # omega = 2.0
    # W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma, omega=omega, k=ang_k, eps=eps)
    # # W = torch.clamp(W, min=eps, max=1.0)
    # W = (W - W.min()) / (W.max() - W.min())

    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W) * d_dimless.mean()

    W = theta
    # print(f"W: {W}")


    logits = - d_dimless * (1 + lam * W) / (tau_d + eps) 

    # print(f"d_dimless: {d_dimless}")
    # print(f"lam: {lam}")
    # print(f"tau_d: {tau_d}")



    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W)

    if center:
        # 행/열 평균 제거로 안정화(선택)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle_mean_5(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    lam_angle: float = 1.0,
    eps: float = 1e-8,
    ang_sigma=0.8,
    ang_k=8.0,
):
    # 기존 시그니처 유지. 내부는 product-of-experts 로짓 사용
    return dist_angle_product_logits_mean_5(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        # ang_mode="softcone",
        ang_mode="gauss",
        # ang_sigma=ang_sigma,
        ang_sigma=2.0,

        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        center=True,
        eps=eps,
    )



def dist_angle_product_logits_mean_6(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    # oxy_sym : 포함/엔테일먼트(콘) 해석을 강조, 반지름 차이가 크고, 로컬 방향 정렬이 중요한 데이터
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    # "softcone": 콘 안팎을 명확히 구분하면서도 바깥에서도 미분 신호 유지, 엔테일먼트/부분순서처럼 “방향 제약”을 강하게 주고 싶을 때, negative가 콘 바깥으로 널리 퍼지는 과제
    # "gauss": 노이즈에 강함, 어디서나 그라디언트 매끄러움, 라벨/정렬 노이즈가 큰 데이터, 초기 학습 안정화(커리큘럼 시작점)
    ang_sigma: float = 0.8,      # 각도 폭(라디안). softcone일 땐 반각 역할
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치(지수)
    center: bool = False,        # 로짓 안정화용 중심화
    eps: float = 1e-8,
):
    """
    sim(x,y) = exp(-√κ d / τ_d) * [ W(θ) ]^λ
    logits   = log sim = -(√κ d)/τ_d + λ * log W(θ)
    -> 가까울수록↑, θ 작을수록↑, 콘 밖에서도 연속적으로↓ (W>0)
    """
    # 하이퍼볼릭 거리 (B1,B2)
    d = pairwise_dist(x, y, curv, eps)               # (단위: 1/√κ)
    d_dimless = (curv**0.5) * d                      # 무차원화

    # 각도 선택
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle_6(x, y, curv, eps)   # (B1,B2)
    elif theta_mode == "origin":
        # 원점 기준 방향 차이(필요시)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1+eps, 1-eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    x_norm = x.norm().clamp_min(eps)
    y_norm = y.norm().clamp_min(eps)
    # lam = (x_norm + y_norm) / 30
    lam = (x_norm + y_norm) / 30

    # 각도 가중치 (0,1]
    # print(f"theta: {theta}")
    # print(f"lam: {lam}")
    
    # omega = 2.0
    # W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma, omega=omega, k=ang_k, eps=eps)
    # # W = torch.clamp(W, min=eps, max=1.0)
    # W = (W - W.min()) / (W.max() - W.min())

    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W) * d_dimless.mean()

    W = theta
    # print(f"W: {W}")


    logits = - d_dimless * (1 + lam * W) / (tau_d + eps) 

    # print(f"d_dimless: {d_dimless}")
    # print(f"lam: {lam}")
    # print(f"tau_d: {tau_d}")



    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W)

    if center:
        # 행/열 평균 제거로 안정화(선택)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle_mean_6(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    lam_angle: float = 1.0,
    eps: float = 1e-8,
    ang_sigma=0.8,
    ang_k=8.0,
):
    # 기존 시그니처 유지. 내부는 product-of-experts 로짓 사용
    return dist_angle_product_logits_mean_6(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        # ang_mode="softcone",
        ang_mode="gauss",
        # ang_sigma=ang_sigma,
        ang_sigma=2.0,

        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        center=True,
        eps=eps,
    )





def dist_angle_product_logits_mean_8(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    # oxy_sym : 포함/엔테일먼트(콘) 해석을 강조, 반지름 차이가 크고, 로컬 방향 정렬이 중요한 데이터
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    # "softcone": 콘 안팎을 명확히 구분하면서도 바깥에서도 미분 신호 유지, 엔테일먼트/부분순서처럼 “방향 제약”을 강하게 주고 싶을 때, negative가 콘 바깥으로 널리 퍼지는 과제
    # "gauss": 노이즈에 강함, 어디서나 그라디언트 매끄러움, 라벨/정렬 노이즈가 큰 데이터, 초기 학습 안정화(커리큘럼 시작점)
    ang_sigma: float = 0.8,      # 각도 폭(라디안). softcone일 땐 반각 역할
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치(지수)
    center: bool = False,        # 로짓 안정화용 중심화
    eps: float = 1e-8,
):
    """
    sim(x,y) = exp(-√κ d / τ_d) * [ W(θ) ]^λ
    logits   = log sim = -(√κ d)/τ_d + λ * log W(θ)
    -> 가까울수록↑, θ 작을수록↑, 콘 밖에서도 연속적으로↓ (W>0)
    """
    # 하이퍼볼릭 거리 (B1,B2)
    d = pairwise_dist(x, y, curv, eps)               # (단위: 1/√κ)
    d_dimless = (curv**0.5) * d                      # 무차원화

    # 각도 선택
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle_6(x, y, curv, eps)   # (B1,B2)
    elif theta_mode == "origin":
        # 원점 기준 방향 차이(필요시)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1+eps, 1-eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    x_norm = x.norm().clamp_min(eps)
    y_norm = y.norm().clamp_min(eps)
    # lam = (x_norm + y_norm) / 30
    lam = (x_norm + y_norm) / 30

    # 각도 가중치 (0,1]
    # print(f"theta: {theta}")
    # print(f"lam: {lam}")
    
    # omega = 2.0
    # W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma, omega=omega, k=ang_k, eps=eps)
    # # W = torch.clamp(W, min=eps, max=1.0)
    # W = (W - W.min()) / (W.max() - W.min())

    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W) * d_dimless.mean()

    W = theta
    # print(f"W: {W}")


    # logits = - d_dimless * (1 + lam * W) / (tau_d + eps) 

    # print(f"d_dimless: {d_dimless}")
    # print(f"lam: {lam}")
    # print(f"tau_d: {tau_d}")



    logits = - d_dimless / (tau_d + eps) + lam * W * d_dimless.mean()

    if center:
        # 행/열 평균 제거로 안정화(선택)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle_mean_8(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    lam_angle: float = 1.0,
    eps: float = 1e-8,
    ang_sigma=0.8,
    ang_k=8.0,
):
    # 기존 시그니처 유지. 내부는 product-of-experts 로짓 사용
    return dist_angle_product_logits_mean_8(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        # ang_mode="softcone",
        ang_mode="gauss",
        # ang_sigma=ang_sigma,
        ang_sigma=2.0,

        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        center=True,
        eps=eps,
    )




def dist_angle_product_logits_mean_9(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    # oxy_sym : 포함/엔테일먼트(콘) 해석을 강조, 반지름 차이가 크고, 로컬 방향 정렬이 중요한 데이터
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    # "softcone": 콘 안팎을 명확히 구분하면서도 바깥에서도 미분 신호 유지, 엔테일먼트/부분순서처럼 “방향 제약”을 강하게 주고 싶을 때, negative가 콘 바깥으로 널리 퍼지는 과제
    # "gauss": 노이즈에 강함, 어디서나 그라디언트 매끄러움, 라벨/정렬 노이즈가 큰 데이터, 초기 학습 안정화(커리큘럼 시작점)
    ang_sigma: float = 0.8,      # 각도 폭(라디안). softcone일 땐 반각 역할
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치(지수)
    center: bool = False,        # 로짓 안정화용 중심화
    eps: float = 1e-8,
):
    """
    sim(x,y) = exp(-√κ d / τ_d) * [ W(θ) ]^λ
    logits   = log sim = -(√κ d)/τ_d + λ * log W(θ)
    -> 가까울수록↑, θ 작을수록↑, 콘 밖에서도 연속적으로↓ (W>0)
    """
    # 하이퍼볼릭 거리 (B1,B2)
    d = pairwise_dist(x, y, curv, eps)               # (단위: 1/√κ)
    d_dimless = (curv**0.5) * d                      # 무차원화

    # 각도 선택
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle_9(x, y, curv, eps)   # (B1,B2)
    elif theta_mode == "origin":
        # 원점 기준 방향 차이(필요시)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1+eps, 1-eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    x_norm = x.norm().clamp_min(eps)
    y_norm = y.norm().clamp_min(eps)
    # lam = (x_norm + y_norm) / 30
    lam = (x_norm + y_norm) / 30

    # 각도 가중치 (0,1]
    # print(f"theta: {theta}")
    # print(f"lam: {lam}")
    
    # omega = 2.0
    # W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma, omega=omega, k=ang_k, eps=eps)
    # # W = torch.clamp(W, min=eps, max=1.0)
    # W = (W - W.min()) / (W.max() - W.min())

    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W) * d_dimless.mean()

    W = theta
    # print(f"W: {W}")


    logits = - d_dimless * (1 + lam * W) / (tau_d + eps) 

    # print(f"d_dimless: {d_dimless}")
    # print(f"lam: {lam}")
    # print(f"tau_d: {tau_d}")



    # logits = - d_dimless / (tau_d + eps) + lam * W * d_dimless.mean()

    if center:
        # 행/열 평균 제거로 안정화(선택)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle_mean_9(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    lam_angle: float = 1.0,
    eps: float = 1e-8,
    ang_sigma=0.8,
    ang_k=8.0,
):
    # 기존 시그니처 유지. 내부는 product-of-experts 로짓 사용
    return dist_angle_product_logits_mean_9(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        # ang_mode="softcone",
        ang_mode="gauss",
        # ang_sigma=ang_sigma,
        ang_sigma=2.0,

        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        center=True,
        eps=eps,
    )



def dist_angle_product_logits_mean_10(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    tau_d: float = 1.0,          # 거리 온도
    theta_mode: str = "oxy_sym", # "oxy_sym" | "origin"
    # oxy_sym : 포함/엔테일먼트(콘) 해석을 강조, 반지름 차이가 크고, 로컬 방향 정렬이 중요한 데이터
    ang_mode: str = "softcone",  # "softcone" | "gauss"
    # "softcone": 콘 안팎을 명확히 구분하면서도 바깥에서도 미분 신호 유지, 엔테일먼트/부분순서처럼 “방향 제약”을 강하게 주고 싶을 때, negative가 콘 바깥으로 널리 퍼지는 과제
    # "gauss": 노이즈에 강함, 어디서나 그라디언트 매끄러움, 라벨/정렬 노이즈가 큰 데이터, 초기 학습 안정화(커리큘럼 시작점)
    ang_sigma: float = 0.8,      # 각도 폭(라디안). softcone일 땐 반각 역할
    ang_k: float = 8.0,          # softcone 경사
    lam: float = 1.0,            # 각도 가중치(지수)
    center: bool = False,        # 로짓 안정화용 중심화
    eps: float = 1e-8,
):
    """
    sim(x,y) = exp(-√κ d / τ_d) * [ W(θ) ]^λ
    logits   = log sim = -(√κ d)/τ_d + λ * log W(θ)
    -> 가까울수록↑, θ 작을수록↑, 콘 밖에서도 연속적으로↓ (W>0)
    """
    # 하이퍼볼릭 거리 (B1,B2)
    d = pairwise_dist(x, y, curv, eps)               # (단위: 1/√κ)
    d_dimless = (curv**0.5) * d                      # 무차원화

    # 각도 선택
    if theta_mode == "oxy_sym":
        theta = symmetric_oxy_angle_10(x, y, curv, eps)   # (B1,B2)
    elif theta_mode == "origin":
        # 원점 기준 방향 차이(필요시)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_th = (x @ y.T) / (x_norm * y_norm.T)
        theta = torch.acos(torch.clamp(cos_th, -1+eps, 1-eps))
    else:
        raise ValueError("theta_mode must be 'oxy_sym' or 'origin'.")

    x_norm = x.norm().clamp_min(eps)
    y_norm = y.norm().clamp_min(eps)
    # lam = (x_norm + y_norm) / 30
    lam = (x_norm + y_norm) / 50

    # 각도 가중치 (0,1]
    # print(f"theta: {theta}")
    # print(f"lam: {lam}")
    
    # omega = 2.0
    # W = angular_weight(theta, mode=ang_mode, sigma=ang_sigma, omega=omega, k=ang_k, eps=eps)
    # # W = torch.clamp(W, min=eps, max=1.0)
    # W = (W - W.min()) / (W.max() - W.min())

    # logits = - d_dimless / (tau_d + eps) + lam * torch.log(W) * d_dimless.mean()

    W = theta
    # print(f"W: {W}")


    logits = - d_dimless * (1 + lam * W) / (tau_d + eps) 

    # print(f"d_dimless: {d_dimless}")
    # print(f"lam: {lam}")
    # print(f"tau_d: {tau_d}")



    # logits = - d_dimless / (tau_d + eps) + lam * W * d_dimless.mean()

    if center:
        # 행/열 평균 제거로 안정화(선택)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits - logits.mean(dim=0, keepdim=True)

    return logits


def joint_logits_dist_angle_mean_10(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor,
    lam_angle: float = 1.0,
    eps: float = 1e-8,
    ang_sigma=0.8,
    ang_k=8.0,
):
    # 기존 시그니처 유지. 내부는 product-of-experts 로짓 사용
    return dist_angle_product_logits_mean_10(
        x, y, curv,
        tau_d=1.0,
        theta_mode="oxy_sym",
        # ang_mode="softcone",
        ang_mode="gauss",
        # ang_sigma=ang_sigma,
        ang_sigma=2.0,

        ang_k=ang_k,
        lam=max(1e-6, float(lam_angle)),
        center=True,
        eps=eps,
    )


def pairwise_oxy_angle(
            x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):

    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1))

    c_xyl = curv * pairwise_inner(x, y, curv)

    acos_numer = y_time[None, :] + c_xyl * x_time[:,None]

    acos_denom = torch.sqrt(torch.clamp(c_xyl ** 2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1)[..., None] * acos_denom + eps)
    _angles = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angles