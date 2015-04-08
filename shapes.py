#!/usr/bin/python

from dxfwrite import DXFEngine as dxf
import numpy as np
from math import *
import mayavi.mlab as mlab

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

## A shape is made of a support curve and a number of generatrices
## represented as vectors starting from the curve
class Shape():
  def __init__(self):
    self.origin = np.array([[0,0,0]]).transpose()
    self.support = np.zeros((3,n_gen))
    self.gen = np.zeros((3,n_gen))

  def rotate(self, matrix):
    self.origin = np.dot(matrix, self.origin)
    self.support = np.dot(matrix, self.support)
    self.gen = np.dot(matrix, self.gen)

class Cone(Shape):
  def __init__(self, angle, length):
    self.offset = np.zeros((3,n_gen+1))
    self.origin = np.array([[0,0,0]]).transpose()
    first = np.array([length, 0, 0])
    first = np.dot(Ry(angle), first)
    self.gen = np.zeros((3,n_gen+1))
    for i in np.arange(0, n_gen+1):
      self.gen[:,i] = np.dot(Rxn(i), first);

  def plot(self):
    dots = self.offset + self.gen
    dots = np.append(self.origin, dots, axis=1)
    #ax = fig.add_subplot(111, projection='3d')
    triangles = [(0, i, i + 1) for i in range(1, n_gen)]
    mlab.triangular_mesh(dots[0,:], dots[1,:], dots[2,:], triangles)

class Cylinder(Shape):
  def __init__(self, radius,length):
    self.origin = np.array([[0,0,0]]).transpose()
    self.support = np.zeros((3, n_gen+1))
    first = np.array([0, radius, 0])
    for i in np.arange(0, n_gen+1):
       self.support[:,i] = np.dot(Rxn(i), first) 
    self.gen = np.zeros((3, n_gen+1))
    for i in np.arange(0, n_gen+1):
      self.gen[:,i] = np.array([length, 0, 0])

  def plot(self):
    dots = self.support + self.gen
    dots = np.append(self.support, dots, axis=1)
    triangles = np.zeros((2*(n_gen+1), 3))
    for i in np.arange(0, n_gen):
        triangles[i,:] = np.array([i, i + n_gen, i + n_gen + 1])
        triangles[i + n_gen,:] = np.array([i, i + 1, i + n_gen + 1])
    print(triangles.shape)
    mlab.triangular_mesh(dots[0,:], dots[1,:], dots[2,:], triangles)

c = Cylinder(1, 10)
c.rotate(Ry(rad(90)))
c.plot()

c = Cone(rad(30), 10)
c.plot()

mlab.show()

