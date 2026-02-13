import numpy as np
import scipy

class Grid:
    def __init__(self, N, config, density, game):
        self.N = N
        self.config = config
        self.density = density # density of live cells
        self.game = game

        # GoL shifts for np.roll
        self.eight_shifts = [(-1, -1), # top-left
                             (-1, 0), # top
                             (-1, 1), # top-right
                             (0, -1), # left
                            (0, 1), # right
                            (1, -1), # bottom-left
                            (1, 0), # bottom
                            (1, 1) # bottom-right
                            ]
        
        # SIRS shifts for np.roll
        self.four_shifts = [(-1, 0), # top
                            (0, -1), # left
                            (0, 1), # right
                            (1, 0) # bottom
                            ]

        self.lattice = None

    def initialise_lattice(self):
        """
        Agnostic lattice initialisation for either game. 
        """
        if self.game == 'GoL':
            if self.config == 'random':
                self.lattice = (np.random.random((self.N, self.N)) < self.density).astype(int) # density of live cells
            # can add more configurations for glider etc. 
        elif self.game == 'SIRS':
            self.lattice = np.random.choice([0, 1, 2], size=(self.N, self.N))

    def count_neighbours(self):
        """
        Counts the nearest-neighbour sum, pertinent to the GoL.
        """
        # make an empty lattice that we can store the total for
        total = np.zeros_like(self.lattice)

        for dx, dy in self.eight_shifts:
            total += np.roll(np.roll(self.lattice, dx, axis=0), dy, axis=1)

        return total
