# Almog Meir, 2021
# Genetic algorithm to find "methuselahs" in Conway's Game of Life, using seagull library to simulate.
# The algorithm initialize random population of soups - random pattern in a given bounding box
# with density of around 33%, then choose according to fitness parents to crossover and possibly mutate
# The life board is torus finite board of given size
# The algorithm results with the best pattern found in n generations, showing the animation, pattern, array.
# Statistics from the different generations also provided at the end of each run.

import numpy as np
import seagull as sg
from seagull.lifeforms import Custom
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random


# get_lifespan gets the history array of the current simulation and the number of iterations,
# it returns the iteration where the pattern get stabilized base on simple, common case (static or 2-step period)
# and complex case (larger period) which require more computing but occur more rarely

def get_lifespan(history, iterations, board):
    # First check for simple case where the stabilized pattern is either static or 2-step period (e.g blinkers)
    for x in range(iterations - 1):
        if np.array_equal(history[x], history[x + 1]) or np.array_equal(history[x], history[x + 2]):
            return x
    # For square boards, loop to catch gliders, by checking frames without 1-pixel frame.
    # If 2 frame without the frame are equal, check if frame + next expected frame if this is a glider
    # are the same, and if so return lifespan of x + 1 period of the glider.
    # This is based on the idea that on square board, a glider period that is not disturbed will complete in
    # 4 times the board length/height
    if board[0] == board[1]:
        for x in range(iterations - 1):
            if np.array_equal(history[x][1:][1:], history[x + 1][1:][1:]) or np.array_equal(history[x][1:][1:],
                                                                                            history[x + 2][1:][1:]):
                next_g = board[0] * 4
                if x + next_g <= iterations and \
                        (np.array_equal(history[x], history[x + next_g]) or
                         np.array_equal(history[x], history[x + next_g + 1])):
                    return int(x + (next_g/2))
    # In case there is no index matching the simple case nor glider case, try to find larger period.
    # The method I decided to use is to calculate the correlation coefficient matrix of the frames
    # and look for the first occurrence of correlation 1 that is not in the diagonal of the matrix
    # which mean for index (x,y) that step x and step y are identical, there for this is a periodic pattern
    flat_history = np.empty((iterations + 1, board[0] * board[1]))
    print("Neither glider or static/2step, creating correlation table")
    for x in range(iterations + 1):
        flat_history[x] = history[x].flatten()  # Flat the matrix to 1d array
    coff_mat = np.corrcoef(flat_history)  # Create correlation matrix
    for index, x in np.ndenumerate(coff_mat):
        if (index[0] != index[1]) and x == 1:  # First case of perfect correlation, return the index
            return index[0]
    # In case there are no 2 identical frames, the pattern was not stabilized under the iterations limit
    return iterations


# random_soup generates a random soup & board of the provided sizes as 2d arrays,
# locating the soup in the middle of the board.
# returns dict object with the board & soup

def random_soup(board_size, soup_size):
    board = sg.Board(size=board_size)
    soup = np.random.randint(3, size=soup_size)     # randomly add values 0,1,2
    soup = np.where(soup == 2, 0, soup)             # change all 2 to 0 to reduce density to ~33%
    middle_board = (int(board_size[0] / 2 - soup_size[0] / 2), int(board_size[1] / 2 - soup_size[1] / 2))
    board.add(Custom(soup), loc=middle_board)
    return {'board': board, 'soup': soup}


# generate_soup is the non-random version of random_soup. working with provided soup as input
# returns the soup & board in dict object

def generate_soup(board_size, soup_size, soup):
    board = sg.Board(size=board_size)
    middle_board = (int(board_size[0] / 2 - soup_size[0] / 2), int(board_size[1] / 2 - soup_size[1] / 2))
    board.add(Custom(soup), loc=middle_board)
    # board.add(sg.lifeforms.gliders.Glider(), loc=(int(BOARD_SIZE[0] / 2), int(BOARD_SIZE[1] / 2)))

    return {'board': board, 'soup': soup}


# initialize_population generate random population of provided size n,
# with provided board size and soup as tuples

def initialize_population(n, board_size, soup_size):
    population_list = [dict() for x in range(n)]
    for x in range(n):
        population_list[x] = random_soup(board_size, soup_size)
    return population_list


# calculate_fitness receives current population as list, and
# calculate the fitness of each pattern in the population by running
# the simulations, summing up the total peaks & lifespans.
# returns a weighted list of corresponding probability to each pattern according
# to the fitness. I chose to work with 60/40 lifespan/peak weight

def calculate_fitness(population, iters_num, board_size, generations_info):
    sum_lifespan = 0
    sum_peak = 0
    run_list = [dict() for x in range(len(population))]
    fitness_arr = np.zeros(len(population))
    generations_info["highest_lifespan"] = 0
    for x in range(len(population)):
        sim = sg.Simulator(population[x]["board"])
        stats = sim.run(sg.rules.conway_classic, iters=iters_num)
        history = sim.get_history()
        run_list[x]["lifespan"] = get_lifespan(history, iters_num, board_size)
        run_list[x]["peak"] = stats["peak_cell_coverage"]
        print("lifespan:", run_list[x]["lifespan"], " , peak:", run_list[x]["peak"])
        generations_info["highest_lifespan"] = max(generations_info["highest_lifespan"], run_list[x]["lifespan"])   #
    for x in run_list:
        sum_lifespan += x["lifespan"]
        sum_peak += x["peak"]
    for x in range(len(run_list)):
        fitness_arr[x] = (run_list[x]["lifespan"]/sum_lifespan*0.6)+(run_list[x]["peak"]/sum_peak*0.4)
    generations_info["avg_lifespan"] = sum_lifespan/len(population)
    generations_info["avg_peak"] = sum_peak / len(population) * board_size[0] * board_size[1]
    return fitness_arr


# crossover receive 2 parents and choose randomly if to cut by rows or columns,
# taking randomly cut_r/c rows/columns from parent1 and n-cut_r/c from parent 2

def crossover(parent1, parent2, soup_size):
    child = np.empty(soup_size)
    if random.random() > 0.5:
        cut_r = np.random.randint(1, soup_size[0])
        child = np.concatenate((parent1[:cut_r, :], parent2[cut_r:, :]))
    else:
        cut_c = np.random.randint(1, soup_size[1])
        child = np.concatenate((parent1[:, :cut_c], parent2[:, cut_c:]), axis=1)
    return child


# mutate takes a soup and randomly flip 1 to 10 bits in the matrix (if 1 then 0 and vice versa)
# returns the mutated soup

def mutate(soup, soup_size):
    n = np.random.randint(1, 11)
    for x in range(n):
        point = (np.random.randint(0, soup_size[0]), np.random.randint(0, soup_size[1]))
        soup[point] = not soup[point]
    return soup


# next_generation receives current population and generates next population by choosing
# parents to cross each time, and mutate the child by probability of 40%
# I decided to keep the best pattern between generations, and I also make sure that
# the same parent can't be chosen for crossover to avoid early convergence.

def next_generation(population, weighted_prob, board_size, soup_size):
    soup_list = np.empty((len(population), soup_size[0], soup_size[1]))
    max_weight = 0
    max_index = 0
    for y in range(len(weighted_prob)):         # Loop to keep best pattern from current population in [0]
        if weighted_prob[y] >= max_weight:
            max_weight = weighted_prob[y]
            max_index = y
    soup_list[0] = population[max_index]["soup"]
    for x in range(1, len(soup_list)):
        parent1 = random.choices(population, weights=weighted_prob)
        parent2 = random.choices(population, weights=weighted_prob)
        while np.array_equal(parent2[0]["soup"], parent1[0]["soup"]):       # make sure to choose 2 different parents
            parent2 = random.choices(population, weights=weighted_prob)
        soup_list[x] = crossover(parent1[0]["soup"], parent2[0]["soup"], soup_size)
        if random.random() > 0.2:
            soup_list[x] = mutate(soup_list[x], soup_size)
    population_list = [dict() for x in range(len(soup_list))]
    population_list[0] = population[max_index]
    for x in range(1, len(soup_list)):
        population_list[x] = generate_soup(board_size, soup_size, soup_list[x])
    return population_list


def main():
    # all values can be changed here
    board_size = (100, 100)
    soup_size = (16, 16)
    iters_num = 15000
    population_size = 15
    num_of_generations = 15

    generations_info = [dict() for x in range(num_of_generations)]      # for saving averages of each generation
    print("Gen 0")
    population = initialize_population(population_size, board_size, soup_size)
    weighted_fitness = calculate_fitness(population, iters_num, board_size, generations_info[0])
    for x in range(1, num_of_generations):
        print("Gen ", x)
        population = next_generation(population, weighted_fitness, board_size, soup_size)
        weighted_fitness = calculate_fitness(population, iters_num, board_size, generations_info[x])
    for x in generations_info:
        print(x)

    x = np.arange(0, num_of_generations, 1)     # x-axis for the plot
    y = np.zeros((num_of_generations, 3))       # y axis for the plot
    for i in range(len(y)):                     # extarct the averages from the dict array
        y[i][0] = generations_info[i]["avg_lifespan"]
        y[i][1] = generations_info[i]["avg_peak"]
        y[i][2] = generations_info[i]["highest_lifespan"]

    # arrange the plot
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o'':', label=("lifespan", "peak", "highest lifespan"))       # Dotted plot with
    fig.legend(title="average:")
    ax.set_title("Averages through generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Amount")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # get the best pattern form last population
    max_weight = 0
    max_index = 0
    for x in range(len(weighted_fitness)):
        if weighted_fitness[x] > max_weight:
            max_weight = weighted_fitness[x]
            max_index = x

    # Simulate, animate and show stats of the chosen pattern
    middle_board = (int(board_size[0] / 2 - soup_size[0] / 2), int(board_size[1] / 2 - soup_size[1] / 2))
    board = sg.Board(size=board_size)
    board.add(Custom(population[max_index]["soup"]), loc=middle_board)
    img = board.view()
    sim = sg.Simulator(board)
    stats = sim.run(sg.rules.conway_classic, iters=iters_num)
    history = sim.get_history()
    best_lifespan = get_lifespan(history, iters_num, board_size)
    best_peak = int(stats["peak_cell_coverage"] * board_size[0] * board_size[1])
    print("Chosen methuselah life span is: ", best_lifespan)
    print("Chosen methuselah peak is: ", best_peak)
    print("Methuselah soup as 2d array:\n", population[max_index]["soup"])
    anim = sim.animate(interval=10)

    plt.show()


if __name__ == '__main__':
    main()
