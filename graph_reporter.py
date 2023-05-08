import matplotlib.pyplot as plt
import neat


class GraphReporter(neat.reporting.BaseReporter):
    fitnesses: [float] = []
    problem_name: str = ""

    def __init__(self, problem_name: str = ""):
        self.problem_name: str = problem_name

    def post_evaluate(self, config, population, species, best_genome):
        if len(self.fitnesses) == 0 or best_genome.fitness > self.fitnesses[-1]:
            self.fitnesses.append(best_genome.fitness)
        else:
            self.fitnesses.append(self.fitnesses[-1])

    def draw_graph(self):
        _, ax = plt.subplots()
        line = ax.plot(self.fitnesses)
        plt.setp(line, color='#7F52FF', linewidth=2.0)
        plt.ylabel('fitness')
        plt.xlabel('generations')
        plt.title(self.problem_name + ". Best fitness: " + str(max(self.fitnesses)))
        plt.show()
