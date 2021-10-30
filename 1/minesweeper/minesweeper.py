import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """

        mines_set = set()
        if len(self.cells) == self.count:
            for cell in self.cells:
                mines_set.add(cell)
        return mines_set
        
        raise NotImplementedError

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """

        safes_set = set()
        if self.count == 0:
            for cell in self.cells:
                safes_set.add(cell)
        return safes_set
        
        raise NotImplementedError

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

#        if self.known_safes():
#            for cell in self.cells:
#                self.mark_safe(cell)
        return
            
        raise NotImplementedError

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)

#        if self.known_mines():
#            for cell in self.cells:
#                self.mark_mine(cell)
        return
        
        raise NotImplementedError


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)

        # 2) mark the cell as safe
        self.safes.add(cell)

        # 3) add a new sentence to the AI's knowledge base
        #   based on the value of `cell` and `count`
        nearby_cells = set()
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if not (i == j == 0):
                    if 0<=cell[0]+i<=7 and 0<=cell[1]+j<=7:
                        nearby_cells.add((cell[0]+i,cell[1]+j))
        new_knowledge = Sentence(nearby_cells,count)
        self.knowledge.append(new_knowledge)


        # 4) mark any additional cells as safe or as mines
        #   if it can be concluded based on the AI's knowledge base
        for cell in nearby_cells:
            if cell in self.mines:
                new_knowledge.mark_mine(cell)
            elif cell in self.safes:
                new_knowledge.mark_safe(cell)

        for cell in new_knowledge.known_mines():
            self.mark_mine(cell)
        for cell in new_knowledge.known_safes():
            self.mark_safe(cell)

        # 5) add any new sentences to the AI's knowledge base
        #   if they can be inferred from existing knowledge
        change = 1
        while change != 0: # keep updating the knowledge until no more new knowledge can be generated
            change = 0
            for i in range(len(self.knowledge)):
                for j in range(i, len(self.knowledge)):
                    if i != j:
                        if len(self.knowledge[i].cells) != 0 and self.knowledge[i].cells.issubset(self.knowledge[j].cells):
                            self.knowledge[j].cells.difference_update(self.knowledge[i].cells)
                            self.knowledge[j].count -= self.knowledge[i].count
                            change += 1
        
        for sentence in self.knowledge:
            for cell in sentence.known_mines():
                self.mark_mine(cell)
            for cell in sentence.known_safes():
                self.mark_safe(cell)
                
        try: # eliminate all the useless information in the knowledge, namely "set() = 0"
            while True:
                self.knowledge.remove(Sentence(set(),0))
        except:
            pass
                
        return
        
        raise NotImplementedError

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        
        diff_set = self.safes.difference(self.moves_made)
        if diff_set:
            length = len(diff_set)
            return list(diff_set)[random.randint(0,length-1)]

        return None
        raise NotImplementedError

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        
        # change height and width according to the size of the board
        height = 8
        width = 8
        
        random_set = set() # random_set is the set of all the possible random moves.
        for i in range(height):
            for j in range(width):
                random_set.add((i,j))
        temp = self.moves_made.union(self.mines)
        random_set.difference_update(temp)
        
        if len(random_set) != 0:
            return list(random_set)[random.randint(0, len(random_set)-1)]

        return None
        raise NotImplementedError
