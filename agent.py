#!/usr/bin/env python3
# encoding: utf-8
"""
agent.py
Template for the Machine Learning Project course at KU Leuven (2017-2018)
of Hendrik Blockeel and Wannes Meert.
Copyright (c) 2018 KU Leuven. All rights reserved.
"""
import sys
import argparse
import logging
import asyncio
import websockets
import json
from collections import defaultdict
import random
import skimage
from skimage import io
import numpy as np
from PIL import Image
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
from collections import deque



logger = logging.getLogger(__name__)
games = {}
agentclass = None
REPLAY_MEMORY = 5000
actions={0:"move",
         1:"right",
         2:"left"}
batch_size = 10

class DQN:
    """Example Dots and Boxes agent implementation base class.
    It returns a random next move.
    A Agent object should implement the following methods:
    - __init__
    - add_player
    - register_action
    - next_action
    - end_game
    This class does not necessarily use the best data structures for the
    approach you want to use.
    """
    def __init__(self, player, nb_rows, nb_cols, mode, action='move'):
        """Create Dots and Boxes agent.
        :param player: Player number, 1 or 2
        :param nb_rows: Rows in grid
        :param nb_cols: Columns in grid
        """
        self.player = player
        self.ended = False
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.action = action
        self.mode = mode
        
        self.im1_processed = np.zeros((1,1,15,15))
        self.im2_processed = np.zeros((1,1,15,15))
        self.im3_processed = np.zeros((1,1,15,15))
        self.im4_processed = np.zeros((1,1,15,15))
        
        self.im4 = Image.new('RGB', (15, 15))
        
        self.buildmodel()
        self.time = 0
        self.observe = 0
        #self.episode = 0
        self.D = deque()
        self.previous_score=0
        self.action_index=0
        self.loss = 0
        
        
    def add_episode(self):
        self.episode +=1
    
    def set_reward(self,msg):
        
        self.new_score = msg['players'][self.player-1]['score']
        
        self.reward = self.new_score - self.previous_score
        
        self.previous_score=msg['players'][self.player-1]['score']
        

        #return  0

    def register_transition(self):
        
        '''we need:
            state in t-1,
            state in t,
            action_index in t-1,
            reward in t, which is difference of score between t-1 and t
        '''
        
        print("this is the reward : " + str(self.reward))
        
        print("this is the action taken last round and to be stored : " + str(actions[self.action_index]))
        
        
        self.im3.save("{}_im3.png".format(self.player))
        print("the previous state is saved as im3.png")
        self.im4.save("{}_im4.png".format(self.player))
        print("the new state is saved as im4.png")
        
        self.D.append((self.state[0], self.action_index, self.reward, self.state[1]))
        if len(self.D) > REPLAY_MEMORY:
            self.D.popleft()
    
    def buildmodel(self):
        
        print("Start building the model")
        
        LEARNING_RATE = 1e-4
        ACTIONS = 3 
        
        self.model = Sequential()
        self.model.add(Convolution2D(10, (1, 1), strides=(1, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(10, (1, 1), strides=(1, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(10, (1, 1), strides=(1, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(ACTIONS))
        self.model.add(Activation('linear'))
        
        adam = Adam(lr=LEARNING_RATE)
        self.model.compile(loss='mse',optimizer=adam)
        
        print("We finish building the model")
        

    def msg_to_img(self,msg):
        #takes in the last message and from that updates the frames
        self.stacked_old = np.concatenate([self.im1_processed,self.im2_processed,self.im3_processed,self.im4_processed], axis=1)
        
        self.im3 = self.im4
        
        self.im1_processed = self.im2_processed
        self.im2_processed = self.im3_processed
        self.im3_processed = self.im4_processed
        
        self.location = msg['players'][self.player-1]['location']
        self.im4 = Image.new('RGB', (15, 15))
        
        head = np.array([7,7])
        #add player in green
        self.im4.putpixel((head[0],head[1]),(0,255,0))
        
        #add apples in red
        for apple in msg['apples']:
            apple_location=np.subtract(self.location,apple)
            if abs(apple_location[0]) < 8 and abs(apple_location[1]) < 8:
                self.im4.putpixel((head[0]-apple_location[0],head[1]-apple_location[1]),(255,0,0))
        
        #add other players in blue
        for opponent in msg['players']:
            if opponent['location'] != ["?","?"] and opponent['location'] != self.location:
                opponent_location=np.subtract(self.location,opponent['location'])
                #you have to put this condition because we don't allow the snake to look further than the border, maybe will later
                if abs(opponent_location[0]) < 8 and abs(opponent_location[1]) < 8:
                    self.im4.putpixel((head[0]-opponent_location[0],head[1]-opponent_location[1]),(0,0,255))
       
        
        ''' 
        #add other players in blue
        opponent_location = msg['players'][1]['location']
        if opponent_location != ["?","?"]:
            opponent=np.subtract(self.location,opponent_location)
            if abs(opponent[0]) < 8 and abs(opponent[1]) < 8:
                self.im4.putpixel((head[0]-opponent[0],head[1]-opponent[1]),(0,0,255))
        '''
        
        
        self.im4_processed = np.asarray(self.im4)
        self.im4_processed = skimage.color.rgb2gray(self.im4_processed)
        self.im4_processed = skimage.transform.resize(self.im4_processed,(15,15))
        self.im4_processed = skimage.exposure.rescale_intensity(self.im4_processed, out_range=(0, 255))
        self.im4_processed = self.im4_processed.reshape(1, 1, 15, 15)
        
        self.stacked_new = np.concatenate([self.im1_processed,self.im2_processed,self.im3_processed,self.im4_processed], axis=1)
        
        self.state = [self.stacked_old,self.stacked_new]
        
        return self.state
#        self.im1.save("{}_im1.png".format(self.player))
#        self.im2.save("{}_im2.png".format(self.player))
#        self.im3.save("{}_im3.png".format(self.player))
#        self.im4.save("{}_im4.png".format(self.player))


    def train(self):
        self.time = 0
        
        batch = random.sample(self.D, batch_size)
        
        state_t, action_t, reward_t, state_t1 = zip(*batch)
        
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        Q_sa = self.model.predict(state_t1)
        targets[range(batch_size), action_t] = reward_t + 0.9*np.max(Q_sa, axis=1)
        
        self.loss += self.model.train_on_batch(state_t, targets)

    def next_action(self):
        """Return the next action this agent wants to perform.
        In this example, the function implements a random move. Replace this
        function with your own approach.
        :return: (row, column, orientation)
        """ 
        playername.observe+=1
        
        logger.info("Computing next move (grid={}x{}, player={})"\
                .format(self.nb_rows, self.nb_cols, self.player))    
        
        if playername.observe < 100:
            
            self.action_index = random.randint(0,2)
        
        
        else:
        
            self.pred= self.model.predict(self.state[1])
            
            max = np.argmax(self.pred)
            
            self.action_index = max
        
        
        return actions[self.action_index]


    def end_game(self):
        self.ended = True


## MAIN EVENT LOOP

async def handler(websocket, path):
    logger.info("Start listening")
    # msg = await websocket.recv()
    try:
        #while(playername.episode<3):
        #LOOP
        
            async for msg in websocket:
                logger.info("< {}".format(msg))
                try:
                    msg = json.loads(msg)
                except json.decoder.JSONDecodeError as err:
                    logger.error(err)
                    return False
                answer = None
                
                if msg["type"] == "start":
                    
                    
                    # Initialize game
                    
                    nb_rows, nb_cols = msg["grid"]
                    
                    if msg["player"] == 1:
                        # Start the game
                        nm =  playername.action
                        
                        print('nm = {}'.format(nm))
                        
                        if nm is None:
                            # Game over
                            logger.info("Game over")
                            continue
                        answer = {
                            'type': 'action',
                            'action': nm,
                        }
                    else:
                        # Wait for the opponent
                        answer = None
    
                elif msg["type"] == "action":
                    # An action has been played
                    #nextplayer is related to items and apples shown
                    #nextplayer is the next one to be able to compute its move, register its actions etc
                    if msg["nextplayer"] == playername.player:                    
                        
                        # Compute your move
                        #storing
                        #if len(playername.register_transition) > batch_size:
                        
                        
                        playername.state = playername.msg_to_img(msg)
                        
                        playername.set_reward(msg)
                        
                        
                        #the transition registered is the state in t-1, the state in t, 
                        #the reward it got from going in t and the action it took to go from
                        #state t-1 to state 2
                        playername.register_transition()
                        
                        if playername.time > 30:
                            
                            playername.train()
                        
                        #determines action index in t
                        nm = playername.next_action()
                        
                        
                        playername.time += 1
                        
                        
                        print("this is the move calculated : " + str(nm))
                        
                        
                        if nm is None:
                            # Game over
                            logger.info("Game over")
                            continue
                        answer = {
                            'type': 'action',
                            'action': nm
                        }
                    else:
                        answer = None
    
                elif msg["type"] == "end":
                    
                    # End the game
                    playername.end_game()
                    playername.add_episode()
                    answer = None
                else:
                    logger.error("Unknown message type:\n{}".format(msg))
    
                if answer is not None:
                    print(answer)
                    await websocket.send(json.dumps(answer))
                    logger.info("> {}".format(answer))
    except websockets.exceptions.ConnectionClosed as err:
        logger.info("Connection closed")
    logger.info(playername.episode)
    logger.info("Exit handler")


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


## COMMAND LINE INTERFACE

def main(argv=None):
    #global agentclass so that it can be reached from different files???
    global agentclass
    global playername
    parser = argparse.ArgumentParser(description='Start agent to play the Apples game')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('port', metavar='PORT', type=int, help='Port to use for server')
    parser.add_argument('mode', metavar='MODE',nargs='?', type=str,default='train', help='train or test?')
    parser.add_argument('action',metavar='ACTION',nargs='?', type=str, default='move', help='which action to perform all the time?')
    parser.add_argument('player',metavar='PLAYER', type=int, help='what player is it?')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    agentclass = DQN
    playername = agentclass(args.player, 36, 16, mode = args.mode, action=args.action)
    print(playername.player)
    #create q-learning agent now
    
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
