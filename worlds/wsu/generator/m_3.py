import math
import numpy as np
import os.path

from .cartpoleplusplus import CartPoleBulletEnv


class CartPolePPMock3(CartPoleBulletEnv):

    def __init__(self, difficulty, renders=False):
        super().__init__(renders=renders)

        self.difficulty = difficulty

        return None

    # Mock is blocks move to spot
    def reset_world(self):
        # Reset world (assume is created)
        p = self._p

        # Delete cartpole
        if self.cartpole == -10:
            self.cartpole = p.loadURDF(os.path.join(self.path, 'models', 'ground_cart.urdf'))
        else:
            p.removeBody(self.cartpole)
            self.cartpole = p.loadURDF(os.path.join(self.path, 'models', 'ground_cart.urdf'))

        # This big line sets the spehrical joint on the pole to loose
        p.setJointMotorControlMultiDof(self.cartpole, 1, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1],
                                       targetVelocity=[0, 0, 0], positionGain=0, velocityGain=0.1,
                                       force=[0, 0, 0])

        # Reset cart (technicaly ground object)
        cart_pos = list(self.np_random.uniform(low=-3, high=3, size=(2,))) + [0]
        cart_vel = list(self.np_random.uniform(low=-50, high=50, size=(2,))) + [0]
        p.resetBasePositionAndOrientation(self.cartpole, cart_pos, [0, 0, 0, 1])
        p.applyExternalForce(self.cartpole, 0, cart_vel, (0, 0, 0), p.LINK_FRAME)

        # Reset pole
        randstate = list(self.np_random.uniform(low=-0.05, high=0.05, size=(6,)))
        pole_pos = randstate[0:3] + [1]
        # zero so it doesnt spin like a top :)
        pole_ori = list(randstate[3:5]) + [0]
        p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_pos, targetVelocity=pole_ori)

        # Delete old blocks
        for i in self.blocks:
            p.removeBody(i)

        # Load blocks in
        self.nb_blocks = np.random.randint(3) + 2
        self.blocks = [None] * self.nb_blocks
        for i in range(self.nb_blocks):
            self.blocks[i] = p.loadURDF(os.path.join(self.path, 'models', 'block.urdf'))

        # Set blocks to be bouncy
        for i in self.blocks:
            p.changeDynamics(i, -1, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0)

        # Set block posistions
        min_dist = 1
        cart_pos, _ = p.getBasePositionAndOrientation(self.cartpole)
        cart_pos = np.asarray(cart_pos)
        for i in self.blocks:
            pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
            pos[2] = pos[2] + 5.0
            while np.linalg.norm(cart_pos[0:2] - pos[0:2]) < min_dist:
                pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
                # Z is not centered at 0.0
                pos[2] = pos[2] + 5.0
            p.resetBasePositionAndOrientation(i, pos, [0, 0, 0, 1])

        # Set block velocities
        u1 = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
        u1[2] = u1[2] + 5.0
        force = np.random.rand() * 10.0
        for i in self.blocks:
            pos, ori = p.getBasePositionAndOrientation(i)
            u2 = np.asarray([u1[0] - pos[0], u1[1] - pos[1], u1[2] - pos[2]])
            if np.linalg.norm(u1) == 0.0:
                u2 = np.asarray([np.random.rand() * 9, np.random.rand() * 9, np.random.rand() * 4 + 5])
            else:
                u2 = np.multiply(u2 / np.linalg.norm(u2), force)

            p.resetBaseVelocity(i, [u2[0], u2[1], u2[2]], [0, 0, 0])

        return None
