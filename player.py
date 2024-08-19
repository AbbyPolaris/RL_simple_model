import model
import envi
import time

Q_ , _ ,pi,_ , _ = model.Q_learning(model.env)
state, terminal = model.env.reset()
print("temrinal:",terminal)
lost_counter = 0

while True:
    while not terminal:
        action = pi(state)
        state , _ , terminal = model.env.step(action)
        time.sleep(.1)
        animation = ""
        for i in range(len(model.env.all_states[0])):
            if i == 5 and model.env.adamak_state == 1:
                animation+= "8"
            elif model.env.all_states[0][i] == 1:
                animation += "0"
            elif model.env.all_states[0][i] == 2:
                animation += "#"    
            else:
                animation += " "
        animation += "\n"
        for i in range(len(model.env.all_states[1])):
            if i == 5 and model.env.adamak_state == 0:
                animation+= "8"
            elif model.env.all_states[1][i] == 1:
                animation += "0"
            elif model.env.all_states[1][i] == 2:
                animation += "#"
            else:
                animation += " "
            
        print(animation)
        print("score:", model.env.score,  "lost:",lost_counter)
    lost_counter += 1
    Q_ , _ ,pi,_ , _ = model.Q_learning(model.env, n_episodes = 10000 + 2000*lost_counter)
    state, terminal = model.env.reset()    
    time.sleep(.5)  