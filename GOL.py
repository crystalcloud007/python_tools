'''
A simulation of Conway's Game of Life.
Create by ZHR. 
'''

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize parametres
ON = 255
OFF =0
vals = [ON,OFF]

'''
CORE FUNCTIONS
Initialize grid, update grid
'''
# Returns a grid of N x N random values
def randomGrid(N):
    return np.random.choice(vals, N*N, p=[0.2,0.8]).reshape(N,N)

# Copys grid since we require 8 neighbours for calculation
# LINE BY LINE
def update(frame_num, img, grid, N):
    new_grid = grid.copy()
    for i in range(N):
        for j in range(N):
            total = int((grid[i, (j-1) % N] + grid[i, (j + 1) % N] + 
                         grid[(i - 1) % N, j] + grid[(i + 1) % N, j] + 
                         grid[(i - 1) % N, (j - 1) % N] + grid[(i - 1) % N, (j + 1 ) % N] +
                         grid[(i + 1) % N, (j - 1) % N] + grid[(i + 1) % N, (j + 1)% N]) / 255)

            # Apply Conway's rules
            # Living cell dies when there are more than 3 or less than 2 neighbours left
            # Living cell lives when there are 3 neighbours left
            # Dead cell back to alive when there are 3 neighbours left
            if grid[i,j] == ON:
                if (total < 2) or (total > 3):
                    new_grid[i,j] = OFF
            else:
                if total == 3:
                    new_grid[i,j] = ON
            
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img

'''
AUXILIARY FUNCTIONS
To initialize special patterns
'''
# Adds a glider with top-left cell at (i,j)
def addGlider(i,j,grid):
    glider = np.array([[0,0,255],[255,0,255],[0,255,255]])
    grid[i:i+3,j:j+3] = glider

# box:2*2, gun:7*8, mouth:7*5, 
def addGosperGun(i,j,grid):
    box = np.array([[255,255],[255,255]])
    grid[i+5:i+7, j+1:j+3] = box

    gun = np.array([[0,0,255,255,0,0,0,0],
                    [0,255,0,0,0,255,0,0],
                    [255,0,0,0,0,0,255,0],
                    [255,0,0,0,255,0,255,255],
                    [255,0,0,0,0,0,255,0],
                    [0,255,0,0,0,255,0,0],
                    [0,0,255,255,0,0,0,0]])
    grid[i+3:i+10,j+11:j+19] = gun

    mouth = np.array([[0,0,0,0,255],
                      [0,0,255,0,255],
                      [255,255,0,0,0],
                      [255,255,0,0,0],
                      [255,255,0,0,0],
                      [0,0,255,0,255],
                      [0,0,0,0,255]])
    grid[i+1:i+8,j+21:j+26] = mouth

    grid[i+3:i+5, j+35:j+37] = box


'''
MAIN FUNCTION
'''
def main():
    parser = argparse.ArgumentParser(description='Run Cellular Automata')
    parser.add_argument('--grid-size', dest='N',required=False)
    #parser.add_argument('--mov-file', dest='mov_file',required=False)
    parser.add_argument('--interval', dest='interval', required=False)
    parser.add_argument('--glider',action='store_true',required=False)
    parser.add_argument('--gosper',action='store_true',required=False)
    args = parser.parse_args()

    # Set grid size
    N = 100
    if args.N and int(args.N) > 8:
        N = int(args.N)
    
    # set animation updating interval
    update_interval = 50
    if args.interval:
        update_interval = int(args.interval)
    
    # declare grid
    grid = np.array([])
    if args.glider:
        grid = np.zeros(N*N).reshape(N,N)
        addGlider(1,1,grid)
    elif args.gosper:
        if N <= 37:
            N = 50
        grid = np.zeros(N*N).reshape(N,N)
        addGosperGun(10,3,grid)
    else:
        grid = randomGrid(N)

    


    # Animation
    plt.rcParams['figure.figsize'] = (10,10)
    fig, ax = plt.subplots()
    img = ax.imshow(grid,interpolation='nearest')
    ani = animation.FuncAnimation(fig, update, fargs=(img,grid,N,),
                                  frames = 10,
                                  interval = update_interval,
                                  save_count = 50)
    # if args.mov_file:
    #     ani.save(args.mov_file, fps = 30, extra_args=['-vcodec','libx264'])
    
    plt.show()

if __name__ == '__main__':
    main()