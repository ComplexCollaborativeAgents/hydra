from worlds.wsu.generator.cartpoleplusplus import CartPoleBulletEnv
import os

class CartPoleNoBlocks(CartPoleBulletEnv):

    def __init__(self, difficulty, renders=False):
        super().__init__(renders=renders)

        self.difficulty = difficulty

        return None

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