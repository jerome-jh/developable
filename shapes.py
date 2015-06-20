#!/usr/bin/python

import math
from dxfwrite import DXFEngine as dxf
import numpy as np
from math import *

#from traits.etsconfig.api import ETSConfig 
#ETSConfig.toolkit = 'qt4'

def rad(d):
  return pi * d / 180.

def deg(r):
  return 180. * r / pi

def fatal(msg):
  print("Fatal error: " + msg + ". Exiting!")
  exit(1)

## number of generatrices for shapes
n_gen = 12

## Rotation matrix around x-axis
def Rx(rad):
 return np.array([[1,0,0],[0,cos(rad),-sin(rad)],[0,sin(rad),cos(rad)]]) 

## Rotation matrix around y-axis
def Ry(rad):
 return np.array([[cos(rad),0,sin(rad)],[0,1,0],[-sin(rad),0,cos(rad)]]) 

## Rotation matrix around z-axis
def Rz(rad):
 return np.array([[cos(rad),-sin(rad),0],[sin(rad),cos(rad),0],[0,0,1]]) 

## Returns the nieth rotation matrix around x axis
def Rxn(n):
  angle = n * 2 * pi / (n_gen - 1)
  return Rx(angle)

## Rotation matrix around arbitrary non null vector
def Rv(vector, rad):
  vector = vector / length(vector)
  cos = math.cos(rad)
  sin = math.sin(rad)
  nx = vector[0]
  ny = vector[1]
  nz = vector[2]
  K = np.ndarray([[0, -nz, ny], [nz, 0, -nz], [-ny, nx, 0]])
  K2 = np.dot(K, K)
  return np.eye(3) + sin*K + (1-cos)*K2
 
## Debug rotation matrix
def debug_rn():
  for i in np.arange(0, n_gen):
    print(Rxn(i))

def length(vector):
  return math.sqrt(np.sum(vector*vector))

def polar(vector):
#  print(vector, vector.shape)
  r = length(vector)
  if r != 0:
    zproj = np.copy(vector)
    zproj[2] = 0
    rz = length(zproj)
    if rz != 0:
      phi = math.acos(np.dot(vector / r, zproj / rz))
      xproj = np.copy(zproj)
      xproj[1] = 0
      rx = length(xproj)
      if rx != 0:
        theta = math.acos(np.dot(zproj / rz, xproj / rx))
      else:
        theta = math.copysign(1, vector[1]) * math.pi / 2
    else:
      theta = 0
      phi = math.copysign(1, vector[2]) * math.pi / 2
  else:
    theta = 0
    phi = 0
  return np.array([r, theta, phi])

## A shape is made of a support curve and a number of generatrices
## represented as vectors starting from the curve
class Shape():
  def __init__(self):
    self.origin = np.array([0,0,0])
    self.support = np.zeros((n_gen,3))
    self.gen = np.zeros((n_gen,3))

  def rotate(self, matrix):
    self.origin = np.dot(self.origin, matrix)
    self.support = np.dot(self.support, matrix)
    self.gen = np.dot(self.gen, matrix)

  def translate(self, vector):
    self.origin = self.origin + vector

  def reverse(self):
    for i in np.arange(self.support.shape[0]):
      self.support[i] = self.support[i] + self.gen[i]  
      self.gen[i] = -self.gen[i]

  def develop(self):
    return

  def _cut(self, offset, vector, plane):
    l = Line(offset, vector)
    alpha = alpha_plane_line(plane, l)
    if alpha <= 0:
      vector = 0
    elif alpha < 1:
      vector = alpha * vector
    return offset, vector

  def cut(self, plane):
    for i in np.arange(self.support.shape[0]):
      self.support[i], self.gen[i] = self._cut(self.support[i], self.gen[i], plane)  

  def plot(self):
    dots = np.zeros((2*(n_gen+1), 3))
    dots[0:n_gen+1] = self.support + self.gen
    dots[n_gen+1:] = self.support
    triangles = np.zeros((2*(n_gen+1), 3))
    for i in np.arange(0, n_gen):
        triangles[i] = np.array([i, i + n_gen, i + n_gen + 1])
        triangles[i + n_gen] = np.array([i, i + 1, i + n_gen + 1])
    #print(dots.shape)
    #print(triangles.shape)
    import mayavi.mlab as mlab
    mlab.triangular_mesh(dots[:,0], dots[:,1], dots[:,2], triangles)


class Cone(Shape):
  def __init__(self, angle, length):
    ## Support is useless (all zeros), but keep it for other methods
    self.support = np.zeros((n_gen+1,3))
    self.offset = np.zeros((n_gen+1,3))
    self.origin = np.array([0,0,0])
    first = np.array([length, 0, 0])
    first = np.dot(first, Ry(angle))
    self.gen = np.zeros((n_gen+1,3))
    for i in np.arange(0, n_gen+1):
      self.gen[i] = np.dot(first, Rxn(i));

#  def plot(self):
#    dots = np.zeros((n_gen+2, 3))
#    dots[0] = self.origin
#    dots[1:] = self.offset + self.gen
#    triangles = np.zeros((n_gen, 3))
#    for i in range(1,n_gen):
#        triangles[i] = np.array([0, i, i + 1])
#    #print(dots.shape)
#    #print(triangles.shape)
#    import mayavi.mlab as mlab
#    mlab.triangular_mesh(dots[:,0], dots[:,1], dots[:,2], triangles)


class Cylinder(Shape):
  """
  A cylinder starting at (0,0,0) and extending towards x
  """
  def __init__(self, radius, length):
    self.origin = np.array([0,0,0])
    self.support = np.zeros((n_gen+1,3))
    first = np.array([0, radius, 0])
    for i in np.arange(0, n_gen+1):
       self.support[i] = np.dot(first, Rxn(i)) 
    self.gen = np.zeros((n_gen+1,3))
    for i in np.arange(0, n_gen+1):
      self.gen[i] = np.array([length, 0, 0])

  def plot(self):
    dots = np.zeros((2*(n_gen+1), 3))
    dots[0:n_gen+1] = self.support + self.gen
    dots[n_gen+1:] = self.support
    triangles = np.zeros((2*(n_gen+1), 3))
    for i in np.arange(0, n_gen):
        triangles[i] = np.array([i, i + n_gen, i + n_gen + 1])
        triangles[i + n_gen] = np.array([i, i + 1, i + n_gen + 1])
    #print(dots.shape)
    #print(triangles.shape)
    import mayavi.mlab as mlab
    mlab.triangular_mesh(dots[:,0], dots[:,1], dots[:,2], triangles)


class Spiral(Shape):
  def __init__(self):
    self.origin = np.array([0,0,0])
    

class Plane():
  def __init__(self, dot, norm):
    dot = np.asarray(dot)
    norm = np.asarray(norm)
    assert(dot.shape == (3,))
    assert(norm.shape == (3,))
    self.dot = dot
    self.norm = norm 

  def rotate(self, matrix):
    self.dot = np.dot(self.dot, matrix)
    self.norm = np.dot(self.norm, matrix)

  def translate(self, vector):
    self.dot = self.dot + vector

  def plot(self, width=10):
    ## Plot a square centered on self.dot
    x, y, z = self.dot
    nx, ny, nz = self.norm
    ## v is orthogonal to self.norm, hence part of the plane
    v = np.array([ny + nz, -nx + nz, -nx - ny])
    v = math.sqrt(2) * width * v / length(v)
    #v2 = np.dot(v, Rv(self.norm, np.pi / 2))
    #print(v, v2)
    m = np.array([[1, -0.5, -0.5],
                  [-0.5, 1, -0.5],
                  [-0.5, -0.5, 1]])
    m2 =  np.zeros(m.shape)
    m2[0] = np.multiply(m[0], v)
    m2[1] = np.multiply(m[1], v) 
    m2[2] = np.multiply(m[2], v) 
    #print(m2)
    import mayavi.mlab as mlab
    mlab.quiver3d(x, y, z, nx, ny, nz)
    #mlab.points3d(self.dot[0], self.dot[1], self.dot[2])
    mlab.triangular_mesh(m2[:,0], m2[:,1], m2[:,2], [[0, 1, 2]])
    

class Line():
  def __init__(self, dot, dir):
    dot = np.asarray(dot)
    dir = np.asarray(dir)
    assert(dot.shape == (3,))
    assert(dir.shape == (3,))
    self.dot = dot
    self.dir = dir 


def inter_plane_line(plane, line):
  #alpha = np.dot(plane.dot, plane.norm) - np.dot(line.dot, plane.norm)
  #alpha = np.dot(plane.norm, plane.dot - line.dot)
  #alpha = alpha / np.dot(line.dir, plane.norm)
  return line.dot + alpha_plane_line(plane, line) * line.dir


def alpha_plane_line(plane, line):
  alpha = np.dot(plane.norm, plane.dot - line.dot)
  return alpha / np.dot(line.dir, plane.norm)


def inter(o1, o2):
  return inter_plane_line(o1, o2)

print("Hello")

#quit()

if False == True:
  c = Cone(rad(30), 10)
  p = Plane([2, 0, 0], [1, 0, 0.5])
  c.reverse()
  c.cut(p)
  c.reverse()
  p.dot += np.array([2,0,0])
  p.norm = np.dot(p.norm, Rx(90))
  c.cut(p)
  c.plot()
  #c.plot()

  c = Cylinder(1, 10)
  p = Plane([1,0,0], [1, 1, 1])
  c.cut(p)
  c.rotate(Ry(rad(90)))
  c.plot()
else:
  c = Cylinder(30, 150)
  c.translate([-20, 0, 0])
  #c.plot()

  p = Plane([0,0,0], [1,0,0])
  p.rotate(Rz(rad(45/2)))
  p.translate(110 - 4 - 7)
  p.plot(50)
  c.cut(p)
  c.plot()
  #c.translate(

import mayavi.mlab as mlab
mlab.show()

