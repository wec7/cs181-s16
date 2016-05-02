# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, algo = "TD Value"):
        self.algo = algo
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 1
        self.eps = 0.1
        if algo == "qlearn":
            self.Q = np.zeros((2,SwingyMonkey.screen_width/50+1,SwingyMonkey.screen_height/50+1,5))
            self.k = np.zeros((2,SwingyMonkey.screen_width/50+1,SwingyMonkey.screen_height/50+1,5)) # number of times action a has been taken from state s
            self.iters = 0
            self.mem = [0, 0]
            self.scores = []
            self.best_score = 50
            self.bestQ = None
        else:
            self.last_feat = None
            self.gravity = 2.0
            self.iter = 1.0
            self.gamma = 0.99
            self.alpha = 10.0
            # some constant that's useful based on observed state
            self.hspeed = 25.0
            self.mk_height = 56.0
            self.tree_gap = 200.0
            self.screen_height = 400.
            self.tree_width = -115.0
            self.avg_jump = 15.0
            self.n_jumps = 0.0

            self.mat = [
                [-5, 10, 1.0], # distance
                [0, 7, 1.0 / self.mk_height], # monkey y loc
                [-8, 8, 2.0 / self.mk_height], # tree relative loc
                [-40, 40, 1.0], #velocity
                [1, 4, 1.0], # gravity
            ]
            self.mat = np.array(self.mat,dtype=np.float)
            sz = self.mat[:,1] - self.mat[:,0] + 1
            sz = sz.astype(np.int)
            vmat = np.zeros(sz, dtype=np.float)
            # prefer to stay in the middle when far away
            vmat[-3:, 3:5, :, :, :] = 0.5

            # avoid hitting tree trunks
            # prepare to pass the tree as we are close
            vmat[:8, :, :3, :, :] = -5
            vmat[:8, :, 3, :40, :] = -3
            vmat[:8, :, 13:, :, :] = -5
            vmat[:8, :, 13, 40:, :] = -3
            vmat[:8, :, 6:11, :, :] = 0.8
            vmat[:1, :, 6:11, :, :] = 1

            # set negative value when close to edge
            vmat[:, 0, :, :, :] = -10
            vmat[:, 1, :, :40, :] = -8
            vmat[:, -1:, :, :, :] = -10
            vmat[:, -2, :, 40:, :] = -8
            self.vmat = vmat
            szr = np.hstack((sz,[2]))
            self.rmat = np.zeros(szr,dtype=np.float)
            self.iter = 1.0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_feat = None
        self.gravity = 2.0
        self.epoch += 1

    def feat_normalize(self, feat):
        '''
        Normalize features
        '''
        idx = np.minimum(np.maximum( feat * self.mat[:,2], self.mat[:,0] ), self.mat[:,1]) - self.mat[:,0]
        idx = idx.astype(np.int)
        return idx

    def transition(self, feat):
        # return estimated feats based on different actions
        feat_next = np.zeros((len(feat), 2))
        # after -115, reset

        if( feat[0] <= -4):
            feat_next[0,:] = 15.0
        else:
            feat_next[0,:] = feat[0] - 1.0
        feat_next[1:3,0] = feat[1:3] + feat[3]
        feat_next[3,0] = feat[3] - feat[4]
        feat_next[4, :] = feat[4]

        feat_next[3,1] = self.avg_jump
        feat_next[1:3,1] = feat[1:3] + self.avg_jump + feat[4]
        return feat_next

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.
        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        if self.algo == "qlearn":
            self.iters += 1
            # current state
            D = state['tree']['dist'] / 50
            if D < 0:
                D = 0
            T = (state['tree']['top']-state['monkey']['top']+0) / 50
            V = state['monkey']['vel'] / 20

            if np.abs(V) > 2:
                V = np.sign(V)*2

            def default_action(p=0.5):
                return 1 if npr.rand() < p else 0

            new_action = default_action()
            if not self.last_action == None:
                # previous state
                d = self.last_state['tree']['dist'] / 50
                t = (self.last_state['tree']['top']-self.last_state['monkey']['top']+0) / 50
                v = self.last_state['monkey']['vel'] / 20

                if np.abs(v) > 2:
                    v = np.sign(v)*2

                max_Q = np.max(self.Q[:,D,T,V])
                new_action = 1 if self.Q[1][D,T,V] > self.Q[0][D,T,V] else 0
                
                # epsilon-greedy
                if self.k[new_action][D,T,V] > 0:
                    eps = self.eps / self.k[new_action][D,T,V]
                else:
                    eps = self.eps
                if (npr.rand() < eps):
                    new_action = default_action()

                if self.k[self.last_action][d,t,v]:
                    ALPHA = 1/self.k[self.last_action][d,t,v]
                    self.Q[self.last_action][d,t,v] += ALPHA*(self.last_reward+0.9*max_Q-self.Q[self.last_action][d,t,v])

            self.mem[0] = state['monkey']['top']

            self.last_action = new_action
            self.last_state  = state
            self.k[new_action][D,T,V] += 1
            return new_action

        # record average jump
        if( self.last_action == 1 ):
            self.avg_jump = (self.n_jumps * self.avg_jump + state["monkey"]["vel"] ) / (self.n_jumps+1.0)
            self.n_jumps += 1

        if self.last_state is not None:
            # estimate gravity
            if self.last_action == 0:
                self.gravity = self.last_state["monkey"]["vel"] - state["monkey"]["vel"]

        feat = np.array([state["tree"]["dist"] / self.hspeed,(state["monkey"]["top"]+state["monkey"]["bot"])/2.0,
                (state["monkey"]["top"] + state["monkey"]["bot"]) / 2.0 - (state["tree"]["top"] + state["tree"]["bot"]) / 2.0,
                state["monkey"]["vel"], self.gravity])

        new_state = state

        idx = self.feat_normalize(feat)

        if( self.last_feat is not None ):
            idxp = self.feat_normalize(self.last_feat)
            idxap = np.hstack((idxp,[self.last_action]))
            # record reward
            if( self.last_reward is not None ):
                self.rmat[tuple(idxap)] = self.last_reward
        else:
            idxp = idx

        # Model Based: check where the monkey would be if no jump
        if self.algo == "Model Based":
            if feat[1] <= self.mk_height * 1.5:
                new_action = 1
            elif feat[1] >= self.screen_height - self.mk_height * 1.5:
                new_action = 0
            elif feat[0] > 0:
                y_no_jump = feat[2] + feat[0] * feat[3] - feat[0] * (feat[0]+1.0)/2.0 * feat[4]
                y_jump = feat[2] + self.avg_jump + feat[4] + feat[0] * self.avg_jump - feat[0] * (feat[0]+1.0)/2.0 * feat[4]

                if feat[0] <= 8 and abs(y_no_jump) >abs(y_jump):
                    new_action = 1
                else:
                    new_action = 0
            else:
                new_action = 0 # not jump when inside the three
        elif self.algo == "TD Value":
            # update value function for previous state
            if( self.last_feat is not None ):
                self.vmat[tuple(idxp)] += self.alpha / (self.last_reward+self.gamma * self.vmat[tuple(idx)]-self.vmat[tuple(idxp)])
            # choose action for this round
            # epsilon greedy for exploration
            if( npr.rand() < self.eps / self.iter ):
                new_action = npr.rand() < 0.5
            else:
                feat_next = self.transition(feat)
                idx0 = self.feat_normalize(feat_next[:,0])
                idx1 = self.feat_normalize(feat_next[:,1])
                idxa0 = np.hstack((idx, [0]))
                idxa1 = np.hstack((idx, [1]))
                val0 = self.rmat[tuple(idxa0)] + self.vmat[tuple(idx0)]
                val1 = self.rmat[tuple(idxa1)] + self.vmat[tuple(idx1)]
                if( val0 >= val1 ):
                    new_action = 0
                else:
                    new_action = 1
        else:
            new_action = npr.rand() < 0.1

        self.iter += 1
        self.last_action = new_action
        self.last_state  = new_state
        self.last_feat = feat

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        print "Epoch: %i |" % ii,
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)
        if learner.algo == "qlearn":
            q_filled = float(np.count_nonzero(learner.Q))*100/learner.Q.size
            print 'score: %d |' % swing.score, 'Q: %s' % str(round(q_filled,3)) + "%"
        else:
            print 'score %d' % swing.score

        # Reset the state of the learner.
        learner.reset()        
    return

if __name__ == '__main__':
    # Select agent
    agent = Learner("Model Based")
    # agent = Learner("TD Value")
    # agent = Learner("qlearn")

	# Run game and save history.
    hist = []
    run_games(agent, hist, 200, 1)
    np.save('hist',np.array(hist))