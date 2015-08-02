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
n_gen = 32

## Rotation matrix around x-axis
def Rx(rad):
 return np.array([[1,0,0],[0,math.cos(rad),-math.sin(rad)],[0,math.sin(rad),math.cos(rad)]]) 

## Rotation matrix around y-axis
def Ry(rad):
 return np.array([[math.cos(rad),0,math.sin(rad)],[0,1,0],[-math.sin(rad),0,math.cos(rad)]]) 

## Rotation matrix around z-axis
def Rz(rad):
 return np.array([[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]]) 

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
    a = np.asarray(a, dtype='f64')
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

  def vs(self, vectors):
    """
    Rotate vectors as lines in a Numpy array
    """
    return np.dot(vectors, np.transpose(self.R))
    

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
 

def cos(u,v):
  """
  Return the cosinus angle between two vectors
  """
  return np.dot(u, v) / (length(u) * length(v))


def sin(u,v):
  """
  Return the sinus angle between two vectors
  """
  return np.cross(u, v) / (length(u) * length(v))


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
    self.support = R.vs(self.support)
    self.gen = R.vs(self.gen)
    return self

  def reverse(self):
    for i in np.arange(self.support.shape[0]):
      self.support[i] = self.support[i] + self.gen[i]  
      self.gen[i] = -self.gen[i]
    return self

  def extend(self, a):
    """
    Extend the length of the generatrices by 'a'
    They are extended in one direction if a>0
    and in the opposite direction if a <0
    Origin is unchanged
    """
    dv = np.zeros(self.gen.shape)
    for i in np.arange(dv.shape[0]):
      dv[i] = (float(abs(a))/length(self.gen[i])) * self.gen[i]
    self.gen = self.gen + dv
    if a < 0:
      self.support = self.support - dv
    return self
       
  def extend2(self, a):
    self.extend(a)
    self.extend(-a)
    return self

  def develop(self):
    """
    Develop the 3D shape into 2D
    """
    return
    ## Every flat generatrice is 2 2D points
    flatgen = np.zeros((n_gen, 2, 2))
    flatgen[0,0] = np.zeros(2)
    flatgen[0,1] = np.array([0, length(self.gen[0])])
    ## 1st generatrice is already in same plane as 0st
    u1 = self.support[1] - self.support[0]
    c = cos(self.gen[1], u1)
    s = cos(self.gen[1], u1)
    d1 = np.array(length(u1)*s + length(u1)*c)
    flatgen[1,0] = flatgen[0,0] + d1
    
    for i in np.range(1, n_gen):
        ## v is rotation axis
        v = self.gen[i]
        u1 = self.support[i] - self.support[i-1]
        u2 = self.support[i] + self.gen[i] - self.support[i-1] - self.gen[i-1] 
	## Pick the best triangle for reference plane
    	c1 = np.cross(u1, v)
        c2 = np.cross(v, u2)
        if length(c1) > length(c2):
           u = u1
        else:
           u = u2
        ## Compute angle between gen and v
        c = cos(self.gen[i+1], v)
        alpha = np.acos(c)
        ## alpha is preserved in rotation, such as length of gen
        
    return self

  def _cut(self, line, plane):
    alpha = alpha_plane_line(plane, line)
    vector = line.dir
    if alpha <= 0:
      vector = 0
    elif alpha < 1:
      vector = alpha * vector 
    return vector

  def cut(self, plane):
    for i in np.arange(self.support.shape[0]):
      self.gen[i] = self._cut(Line(self.gen[i], self.origin + self.support[i]), plane)  
    return self

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
  def __init__(self, radius, length, origin=nullvect, axis=[1,0,0]):
    self.origin, self.axis = _toarr(origin, axis)
    self.support = np.zeros((n_gen+1,3))
    first = np.array([0, radius, 0])
    for i in np.arange(0, n_gen+1):
       self.support[i] = np.dot(first, Rxn(i)) 
    self.gen = np.zeros((n_gen+1,3))
    for i in np.arange(0, n_gen+1):
      self.gen[i] = np.array([length, 0, 0])

  def rotate(self, axis, angle, origin=nullvect):
    R = Rotation(axis, angle, origin)
    self.origin = R.p(self.origin)
    self.support = R.vs(self.support)
    self.gen = R.vs(self.gen)
    self.axis = R.v(self.axis)
    return self

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
    v = np.zeros((4,3))
    ## v is orthogonal to self.norm, hence part of the plane
    v[0] = np.array([ny + nz, -nx + nz, -nx - ny])
    v[0] = math.sqrt(2) * width * v[0] / length(v[0])
    v[2] = -v[0]
    R = Rotation(self.norm, np.pi/2)
    v[1] = R.v(v[0])
    v[3] = R.v(v[2])
    v = v + self.origin
    import mayavi.mlab as mlab
    mlab.quiver3d(x, y, z, nx, ny, nz)
    mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], [[0, 1, 2], [2, 3, 0]])
    

class Line(Point):
  def __init__(self, dir, origin=nullvect):
    self.dir, self.origin = _toarr(dir, origin)


def alpha_plane_line(plane, line):
  """
  Compute alpha, which is the scaling factor for the line direction vector
  to reach the plane
  """
  alpha = np.dot(plane.norm, plane.origin - line.origin)
  return alpha / np.dot(line.dir, plane.norm)

def inter_plane_line(plane, line):
  """
  Compute the intersection point between the line and the plane
  """
  return line.origin + alpha_plane_line(plane, line) * line.dir

def inter(o1, o2):
  return inter_plane_line(o1, o2)

print("Hello")

def inv_rot(u, v, a):
  """
  Inverse rotation around vector u, that brings vector a in plane
  defined by u,v
  """
  u, v, a = _toarr(u, v, a)
  n = np.cross(u, v)
  if np.all(n == nullvect):
    return 0
  ux, uy, uz = u
  print(ux, uy, uz)
  vx, vy, vz = v 
  print(vx, vy, vz)
  x, y, z = a
  print(x, y, z)
  return

#quit()

b = Base()
b.plot(10)

R = Rotation(b.x, np.pi/4)
v = [0, 1, 0]
vr = R.v(v)
print(vr)
print(inv_rot(b.x, b.y, vr))

def packraft():
  import copy
  import math

  ## Radius of tubes
  r = 25./2
  ## Inner width
  w = 40
  ## Inner length
  l = 110
  ## Bow upturn
  u = 6
  ## Bow angle
  a = math.atan(u / 37)
  print(a)
  a = rad(30)

  p1 = Plane(b.x)
  p1.rotate(b.z, -np.pi/8)

  p2 = Plane(b.y)
  p2.translate(-10*b.x+10*b.y)
  p2.rotate(b.z, np.pi/8, p2.origin)

  s1 = Cylinder(r, 90)
  s1.translate(-r*b.y)

  c1 = Cylinder(r, 10)
  c1.translate(r*b.y)
  c1.rotate(b.z, 3*np.pi/4)

  b1 = Cylinder(r, 20)
  b1.rotate(b.z, np.pi/2)
  b1.translate(10*b.y - (r+10)*b.x)
  
  p1.rotate(b.x, a, [0, -r, 0])
  c1.rotate(b.x, a, [0, -r, 0])
  p2.rotate(b.x, a, [0, -r, 0])
  b1.rotate(b.x, a, [0, -r, 0])

  i21 = inter_plane_line(p2, Line(c1.axis, c1.origin))
  i22 = inter_plane_line(p2, Line(b1.axis, b1.origin))
  print(i21)
  print(i22)

  b1.rotate(b.x, -a, i21)
  p2.rotate(b.x, -a/2, i21)

  s1.extend2(20)
  c1.extend2(20)
  b1.extend2(20)
  
  s1.reverse().cut(p1)
  c1.reverse().cut(p1)
  c1.reverse().cut(p2)
  b1.reverse().cut(p2)

  #p1.plot(30)
  #p2.plot(30)

  s1.plot()
  c1.plot()
  b1.plot()
  return

  s2 = copy.deepcopy(s1)
  s2.translate(b.y * (w + r))
  s2.plot()

  b1 = Cylinder(r, 90)
  b1.rotate(b.z, np.pi/2)
  b1.translate(-b.x * (10 + r))
  b1.translate(b.z * u)
  b1.plot()

  b2 = copy.deepcopy(b1)
  b2.translate(b.x * (l + r))
  b2.plot()
  
  c1 = Cylinder(r, 90)
  c1.translate(-b.y*r -60*b.x)
  c1.rotate(b.y, a) 
  c1.rotate(b.z, -np.pi/4, c1.origin + 60*b.x)
  c1.plot()
  
  return

  c.reverse()
  p = Plane(b.x)
  p.rotate(b.z, -np.pi/8)
  c.cut(p)
  p.rotate(b.z, np.pi/4)
  p.translate(90*b.x)
  c.reverse()
  c.cut(p)

  s1 = copy.deepcopy(c)
  s1.plot()

  c.rotate(b.x, np.pi)
  c.translate(b.y * (40 + 2*r))
  
  s1 = copy.deepcopy(c)
  s1.plot()

  c = Cylinder(r, 90)
  c.translate(-20*b.x)
  c.reverse()
  p = Plane(b.x)
  p.rotate(b.z, -np.pi/8)
  c.cut(p)
  p.rotate(b.z, np.pi/4)
  p.translate(20*b.x)
  c.reverse()
  c.cut(p)
  c.rotate(b.x, np.pi)
  c.rotate(b.z, np.pi/2)
  c.translate(-10*b.x + b.y*(10-r*np.cos(np.pi/8)))

  c.plot()
  p.plot()

packraft()
import mayavi.mlab as mlab
mlab.show()
quit()

def test_extend():
  import copy
  c = Cylinder(10, 10)
  c.plot()
  c = copy.deepcopy(c)
  c.translate(b.y*20)
  c.extend(10)
  c.plot()
  c = copy.deepcopy(c)
  c.translate(b.y*20)
  c.extend(-20)
  c.plot()
  print(c.origin)
  import mayavi.mlab as mlab
  mlab.show()


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
  p.rotate(b.z, rad(45))
  p.translate((110 - 4 - 7) * b.x)
  p.plot(200)
  c.cut(p)
  c.plot()
  #c.translate(

import mayavi.mlab as mlab
mlab.show()

