import numpy as np
import pygame
from TheGobletOfFire import GridWorldEnvironment  

Q=np.load("q_table.npy")
env=GridWorldEnvironment("15x10",render_mode="human")
#testing using the updated q values
for ep in range(10):
    harry,cup,death_eater=env.reset()
    r_harry,c_harry=harry
    r_death,c_death=death_eater

    done=False
    steps=0
    while not done and steps<=200:
        env.render()

        state_harry=r_harry*env.ncol+c_harry
        state_death=r_death*env.ncol+c_death

        action=np.argmax(Q[state_harry,state_death])
        harry_pos,death_pos=env.traverse(action)
        r_harry,c_harry=harry_pos
        r_death,c_death=death_pos
        if harry_pos==cup:
            print(f"Episode {ep+1}: Harry wins!")
            done=True
        elif harry_pos==death_pos:
            print(f"Episode {ep+1}: Harry caught!")
            done=True

        steps+=1

pygame.quit()
