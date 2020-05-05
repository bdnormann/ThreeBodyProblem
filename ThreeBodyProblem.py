"""
This script is adapted from 
1) The code at http://matplotlib.sourceforge.net/examples/animation/double_pendulum_animated.py
Double pendulum formula, which was translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c
and
2) modified by Jake Vanderplas, email: vanderplas@astro.washington.edu, website: http://jakevdp.github.com, license: BSD

Building on the above, this script is an implementation of the Verlet algorithm for 3 bodies in interaction.
Author: Ben David Normann

Please feel free to use and modify this, keeping the above information. Thanks!
"""


import numpy as np
from matplotlib.pyplot import figure, show
import matplotlib.animation as animation
from math import *


# This is a class describing three planets in interaction. init_state is [x1,y1, v1x,v1y, x2,y2,v2x,v2y, xS, yS,
# vSx, vSy], where x,y denotes initial positions and vx,vy denote initial velocities. S is for sun (heavy object).
class ThreeBodyProblem:

    # This method shows how an instance of the class should be implemented:
    #def __init__(self, init_state=None, sun=100, s1=1, s2=1, origin=(0, 0)):
    def __init__(self, init_state, sun_mass, planet1_mass, planet2_mass, x0,y0):
        # the mass of the Sun and planets s1 and s2 (normalized):
        sun=sun_mass
        s1=planet1_mass
        s2=planet2_mass
        if init_state is None:
            init_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Just making an array out of the input parameters, with type float
        self.init_state = np.asarray(init_state, dtype='float')
        self.params = (sun, s1, s2)
        self.origin = (x0,y0)
        self.time_elapsed = 0
        self.state = self.init_state

    #Now we define two arrays x and y of positions, which will be updated by self.step (to be defined):

    def position(self):
        x = np.asarray([self.state[0],
                        self.state[4],
                        self.state[8]])
        y = np.asarray([self.state[1],
                        self.state[5],
                        self.state[9]])
        return (x, y)

    # Function calculating the energy:
    def energy(self):
        #compute the energy of the current state
        (sun, s1, s2) = self.params

        # positoins
        dx = self.state[0] - self.state[4]
        dy = self.state[1] - self.state[5]
        dr = sqrt(pow(dx, 2) + pow(dy, 2))
        dxs1 = self.state[0] - self.state[8]
        dys1 = self.state[1] - self.state[9]
        drs1 = sqrt(pow(dxs1, 2) + pow(dys1, 2))
        dxs2 = self.state[4] - self.state[8]
        dys2 = self.state[5] - self.state[9]
        drs2 = sqrt(pow(dxs2, 2) + pow(dys2, 2))

        # Energy
        K = 0.5 * (self.state[2] * self.state[2] + self.state[3] * self.state[3]) + 0.5 * (s1 / s2) * (
                self.state[6] * self.state[6] + self.state[7] * self.state[7]) + 0.5 * (sun / s2) * (
                    self.state[10] * self.state[10] + self.state[11] * self.state[
                11])  # Normalized by m1*V^2_normalization
        U = -sun / drs1 - (sun * s1 / s2) / drs2 - s1 / dr  # Normalized by m1*V^2_normalization

        return U + K

    def step(self, dt):
        (sun, s1, s2) = self.params
        # The Verlet algorithm is hereunder implemented:

        # Positions:
        dx = self.state[0] - self.state[4]
        dy = self.state[1] - self.state[5]
        dr = sqrt(pow(dx, 2) + pow(dy, 2))
        dxs1 = self.state[0] - self.state[8]
        dys1 = self.state[1] - self.state[9]
        drs1 = sqrt(pow(dxs1, 2) + pow(dys1, 2))
        dxs2 = self.state[4] - self.state[8]
        dys2 = self.state[5] - self.state[9]
        drs2 = sqrt(pow(dxs2, 2) + pow(dys2, 2))

        # Calculating the sum of forces on planet 1 (f1x,f1y), planet 2 (f2x,f2y) and planet 3 (f3x,f3y):
        f1x = -sun * dxs1 / pow(drs1, 3) - s1 * dx / pow(dr, 3)
        f1y = -sun * dys1 / pow(drs1, 3) - s1 * dy / pow(dr, 3)
        f2x = -sun * dxs2 / pow(drs2, 3) + s2 * dx / pow(dr, 3)
        f2y = -sun * dys2 / pow(drs2, 3) + s2 * dy / pow(dr, 3)
        fSx = s2 * dxs1 / pow(drs1, 3) + s1 * dxs2 / pow(drs2, 3)
        fSy = s2 * dys1 / pow(drs1, 3) + s1 * dys2 / pow(drs2, 3)

        # Calculating v(t+dt/2):
        v1xm = self.state[2] + 0.5 * dt * f1x
        v1ym = self.state[3] + 0.5 * dt * f1y
        v2xm = self.state[6] + 0.5 * dt * f2x
        v2ym = self.state[7] + 0.5 * dt * f2y
        vSxm = self.state[10] + 0.5 * dt * fSx
        vSym = self.state[11] + 0.5 * dt * fSy

        # Calculating r(t+dt):
        self.state[0] = self.state[0] + v1xm * dt
        self.state[1] = self.state[1] + v1ym * dt
        self.state[4] = self.state[4] + v2xm * dt
        self.state[5] = self.state[5] + v2ym * dt
        self.state[8] = self.state[8] + vSxm * dt
        self.state[9] = self.state[9] + vSym * dt

        # Starting over again to calculate v(t+dt):

        # Positions:
        dx = self.state[0] - self.state[4]
        dy = self.state[1] - self.state[5]
        dr = sqrt(pow(dx, 2) + pow(dy, 2))
        dxs1 = self.state[0] - self.state[8]
        dys1 = self.state[1] - self.state[9]
        drs1 = sqrt(pow(dxs1, 2) + pow(dys1, 2))
        dxs2 = self.state[4] - self.state[8]
        dys2 = self.state[5] - self.state[9]
        drs2 = sqrt(pow(dxs2, 2) + pow(dys2, 2))

        # Calculating the sum of forces on planet 1 (f1x,f1y), planet 2 (f2x,f2y) and planet 3 (f3x,f3y):
        f1x = -sun * dxs1 / pow(drs1, 3) - s1 * dx / pow(dr, 3)
        f1y = -sun * dys1 / pow(drs1, 3) - s1 * dy / pow(dr, 3)
        f2x = -sun * dxs2 / pow(drs2, 3) + s2 * dx / pow(dr, 3)
        f2y = -sun * dys2 / pow(drs2, 3) + s2 * dy / pow(dr, 3)
        fSx = s2 * dxs1 / pow(drs1, 3) + s1 * dxs2 / pow(drs2, 3)
        fSy = s2 * dys1 / pow(drs1, 3) + s1 * dys2 / pow(drs2, 3)

        # Calculating v(t+dt):
        self.state[2] = v1xm + 0.5 * dt * f1x
        self.state[3] = v1ym + 0.5 * dt * f1y
        self.state[6] = v2xm + 0.5 * dt * f2x
        self.state[7] = v2ym + 0.5 * dt * f2y
        self.state[10] = vSxm + 0.5 * dt * fSx
        self.state[11] = vSym + 0.5 * dt * fSy

        # Execute one time step of length dt and update state
        self.time_elapsed += dt


# ------------------------------------------------------------
# initialising an instance
threebody = ThreeBodyProblem([-7, 0, 0, 2.5, -6.5, 0, 0, 0.5, 0, 0, 0, 0] , 100 , 1 , 1,0,0)
dt = 1. / 500

# ------------------------------------------------------------
# set up figure and animation
fig = figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-15, 15), ylim=(-15, 15))
# ax.grid()

line, = ax.plot([], [], 'o') #gives the marker
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)


def init():
    #initialize animation
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text


def animate(i):
    #perform animation step
    global threebody, dt
    threebody.step(dt)

    line.set_data(*threebody.position())
    line.set_color('gray')
    time_text.set_text('time = %.1f' % threebody.time_elapsed)
    energy_text.set_text('energy = %.3f ' % threebody.energy())
    return line, time_text, energy_text


# choose the interval based on dt and the time to animate one step
from time import time

t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=1000,
                              interval=interval, blit=True, init_func=init)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# ani.save('OrbitAnim.mp4', fps=500, extra_args=['-vcodec', 'libx264'])

show()
