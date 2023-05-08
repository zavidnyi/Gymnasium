import multiprocessing
import os
import pickle
import numpy as np
import neat
import gymnasium as gym

import graph_reporter

runs_per_net = 2


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config, visualise: bool = False):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        if visualise:
            env = gym.make("MountainCar-v0", render_mode="human")
        else:
            env = gym.make("MountainCar-v0")
        observation = env.reset()[0]
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False
        truncated = False
        while not done and not truncated:
            action = np.argmax(net.activate(observation))
            observation, reward, done, truncated, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config")

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    graph = graph_reporter.GraphReporter("Mountain car")
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(graph)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, 100)

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)
    # eval_genome(winner, config, True)
    graph.draw_graph()
    print(winner)


if __name__ == '__main__':
    run()
