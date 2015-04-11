#!/usr/bin/python

import math
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

  def develop(self):
    return

class Cone(Shape):
  def __init__(self, angle, length):
    self.offset = np.zeros((n_gen+1,3))
    self.origin = np.array([0,0,0])
    first = np.array([length, 0, 0])
    first = np.dot(first, Ry(angle))
    self.gen = np.zeros((n_gen+1,3))
    for i in np.arange(0, n_gen+1):
      self.gen[i] = np.dot(first, Rxn(i));

  def plot(self):
    dots = np.zeros((n_gen+2, 3))
    dots[0] = self.origin
    dots[1:] = self.offset + self.gen
    triangles = np.zeros((n_gen, 3))
    for i in range(1,n_gen):
        triangles[i] = np.array([0, i, i + 1])
    #print(dots.shape)
    #print(triangles.shape)
    mlab.triangular_mesh(dots[:,0], dots[:,1], dots[:,2], triangles)

class Cylinder(Shape):
  def __init__(self, radius,length):
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
    mlab.triangular_mesh(dots[:,0], dots[:,1], dots[:,2], triangles)

class Spiral(Shape):
  def __init__(self):
    self.origin = np.array([0,0,0])
    

c = Cone(rad(30), 10)
c.plot()

c = Cylinder(1, 10)
c.rotate(Ry(rad(90)))
c.plot()

mlab.show()

print(polar(np.array([0,0,0])))
print(polar(np.array([1,0,0])))
print(polar(np.array([0,1,0])))
print(polar(np.array([0,0,1])))
print(polar(np.array([1,0,1])))
print(polar(np.array([1,1,0])))
print(polar(np.array([0,1,1])))
print(polar(np.array([1,1,1])))
