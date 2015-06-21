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

## Debug rotation matrix
def debug_rn():
  for i in np.arange(0, n_gen):
    print(Rxn(i))

def length(vector):
  return math.sqrt(np.sum(np.dot(vector, vector)))

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


nullvect = np.array([0,0,0])


def _toarr(*args):
  ret = list()
  for a in args:
    a = np.asarray(a)
    assert(a.shape == (3,))
    ret.append(a)
  if len(ret) == 1:
    return ret[0]
  else:
    return tuple(ret)


class Rotation():
  """
  Rotation around axis vector passing through origin
  """
  def __init__(self, axis, angle, origin=nullvect):
    self.origin = _toarr(origin)
    ## The rotation matrix
    self.R = self.Rv(axis, angle)

  ## Rotation matrix around arbitrary non null vector
  def Rv(self, axis, angle):
    axis = _toarr(axis)
    axis = axis / length(axis)
    cos = math.cos(angle)
    sin = math.sin(angle)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    axis = np.reshape(axis, (1,3))
    K2 = np.dot(np.transpose(axis), axis)
    return cos * np.eye(3) + sin * K + (1-cos)*K2
 
  def p(self, point):
    """
    Rotate point
    """
    point = _toarr(point)
    if np.all(self.origin == nullvect):
      return self.v(point)
    else:
      return self.v(point) - self.v(self.origin) + self.origin

  def v(self, vect):
    """
    Rotate vector
    """
    vect = _toarr(vect)
    return np.dot(self.R, vect)


class Point():
  """
  The Point class is for display only, not for calculations
  """
  def __init__(self, origin=nullvect):
    self.origin = _toarr(origin)

  def translate(self, vector):
    self.origin = self.origin + _toarr(vector)
    return self

  def rotate(self, axis, angle, origin=nullvect):
    R = Rotation(axis, angle, origin)
    self.origin = R.p(self.origin)
    return self

  def plot(self):
    x, y, z = self.origin
    import mayavi.mlab as mlab
    mlab.points3d(x, y, z)
    return self


class Vector():
  """
  The Vector class is for display only, not for calculations
  Hence it has an origin, which actual vectors in the code do not have
  """
  def __init__(self, v, origin=nullvect):
    self.origin, self.v = _toarr(origin, v)

  def translate(self, vector):
    self.origin = self.origin + _toarr(vector)
    return self

  def rotate(self, axis, angle, origin=nullvect):
    R = Rotation(axis, angle, origin)
    self.origin = R.p(self.origin)
    self.v = R.v(self.v)
    return self

  def plot(self):
    x, y, z = self.origin
    vx, vy, vz = self.v
    import mayavi.mlab as mlab
    mlab.quiver3d(x, y, z, vx, vy, vz, scale_factor=length(self.v))
    return self


class Base():
  def __init__(self):
    self.x = np.array([1, 0, 0])
    self.y = np.array([0, 1, 0])
    self.z = np.array([0, 0, 1])

  def plot(self, length=1):
    Vector(length * self.x).plot()
    Vector(length * self.y).plot()
    Vector(length * self.z).plot()
    return self
    

## A shape is made of a support curve and a number of generatrices
## represented as vectors starting from the curve
class Developable():
  def __init__(self, origin=nullvect):
    self.origin = _toarr(origin)
    self.support = np.zeros((n_gen,3))
    self.gen = np.zeros((n_gen,3))

  def translate(self, vector):
    self.origin = self.origin + _toarr(vector)
    return self

  def rotate(self, axis, angle, origin=nullvect):
    R = Rotation(axis, angle, origin)
    self.origin = R.p(self.origin)
    self.support = np.dot(self.support, np.transpose(R.R))
    self.gen = np.dot(self.gen, np.transpose(R.R))
    return self

  def reverse(self):
    for i in np.arange(self.support.shape[0]):
      self.support[i] = self.support[i] + self.gen[i]  
      self.gen[i] = -self.gen[i]

  def develop(self):
    return

  def _cut(self, offset, vector, plane):
    l = Line(vector, offset)
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
    dots[0:n_gen+1] = self.origin + self.support + self.gen
    dots[n_gen+1:] = self.origin + self.support
    triangles = np.zeros((2*(n_gen+1), 3))
    for i in np.arange(0, n_gen):
        triangles[i] = np.array([i, i + n_gen, i + n_gen + 1])
        triangles[i + n_gen] = np.array([i, i + 1, i + n_gen + 1])
    #print(dots.shape)
    #print(triangles.shape)
    import mayavi.mlab as mlab
    mlab.triangular_mesh(dots[:,0], dots[:,1], dots[:,2], triangles)


class Cone(Developable):
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


class Cylinder(Developable):
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

#  def plot(self):
#    dots = np.zeros((2*(n_gen+1), 3))
#    dots[0:n_gen+1] = self.support + self.gen
#    dots[n_gen+1:] = self.support
#    triangles = np.zeros((2*(n_gen+1), 3))
#    for i in np.arange(0, n_gen):
#        triangles[i] = np.array([i, i + n_gen, i + n_gen + 1])
#        triangles[i + n_gen] = np.array([i, i + 1, i + n_gen + 1])
#    #print(dots.shape)
#    #print(triangles.shape)
#    import mayavi.mlab as mlab
#    mlab.triangular_mesh(dots[:,0], dots[:,1], dots[:,2], triangles)


class Spiral(Developable):
  def __init__(self):
    self.origin = np.array([0,0,0])
    

class Plane(Point):
  def __init__(self, norm, origin=np.array([0, 0, 0])):
    self.origin, self.norm = _toarr(origin, norm)

  def translate(self, vector):
    self.origin = self.origin + _toarr(vector)
    return self

  def rotate(self, axis, angle, origin=nullvect):
    R = Rotation(axis, angle, origin)
    self.origin = R.p(self.origin)
    self.norm = R.v(self.norm)
    return self

  def plot(self, width=10):
    ## Plot a square centered on self.origin
    x, y, z = self.origin
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
    m2[0] = np.multiply(m[0], v) + self.origin
    m2[1] = np.multiply(m[1], v) + self.origin
    m2[2] = np.multiply(m[2], v) + self.origin
    #print(m2)
    import mayavi.mlab as mlab
    mlab.quiver3d(x, y, z, nx, ny, nz)
    mlab.triangular_mesh(m2[:,0], m2[:,1], m2[:,2], [[0, 1, 2]])
    

class Line(Point):
  def __init__(self, dir, origin=np.array([0, 0, 0])):
    origin = np.asarray(origin)
    dir = np.asarray(dir)
    assert(origin.shape == (3,))
    assert(dir.shape == (3,))
    self.origin = origin
    self.dir = dir 


def inter_plane_line(plane, line):
  #alpha = np.dot(plane.origin, plane.norm) - np.dot(line.origin, plane.norm)
  #alpha = np.dot(plane.norm, plane.origin - line.origin)
  #alpha = alpha / np.dot(line.dir, plane.norm)
  return line.origin + alpha_plane_line(plane, line) * line.dir


def alpha_plane_line(plane, line):
  alpha = np.dot(plane.norm, plane.origin - line.origin)
  return alpha / np.dot(line.dir, plane.norm)


def inter(o1, o2):
  return inter_plane_line(o1, o2)

print("Hello")

#quit()

b = Base()
b.plot(10)

if False == True:
  c = Cone(rad(30), 10)
  p = Plane([2, 0, 0], [1, 0, 0.5])
  c.reverse()
  c.cut(p)
  c.reverse()
  p.origin += np.array([2,0,0])
  p.norm = np.dot(p.norm, Rx(90))
  c.cut(p)
  c.plot()
  #c.plot()

  c = Cylinder(1, 10)
  p = Plane([1, 1, 1], [1,0,0])
  c.cut(p)
  c.rotate(Ry(rad(90)))
  c.plot()
elif False == True:
  p = Point()
  p.plot()
  p.translate(2*b.x).plot()
  p.rotate(b.y, np.pi/2).plot()
  print(p.origin)
  p.rotate(b.z, -np.pi/2).plot()
  print(Rotation(b.x, np.pi/2).R)
  print(Rotation(b.y, np.pi/2).R)
  print(Rotation(b.z, np.pi/2).R)
  print(p.origin)
  p = Point(np.array([0.5, 0.5, 0.5])).plot()
  ## Rotating a point around itself, should have no effect
  p.rotate(b.x, np.pi/2, p.origin).plot()
  p.rotate(b.x, 0).plot()
else:
  c = Cylinder(30, 150)
  #c.plot()
  c.translate([50, 0, 0])
  #c.rotate(b.y, np.pi/3, c.origin)
  c.rotate(b.y, np.pi/3)
  #c.plot()

  p = Plane([1,0,0])
  p.rotate(b.z, rad(45/2))
  p.translate((110 - 4 - 7) * b.x)
  p.plot(200)
  c.cut(p)
  c.plot()
  #c.translate(

import mayavi.mlab as mlab
mlab.show()

