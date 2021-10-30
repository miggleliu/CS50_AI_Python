import sys
import math

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        
        for v in self.domains:
            temp = self.domains[v].copy()
            for x in self.domains[v]:
                if len(x) != v.length:
                    temp.remove(x)
            self.domains[v] = temp
        return
        raise NotImplementedError

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        is_revision_made = False
        x_idx = self.crossword.overlaps[x,y][0]
        y_idx = self.crossword.overlaps[x,y][1]
        
        possible_overlaps = set()
        for y_word in self.domains[y]:
            possible_overlaps.add(y_word[y_idx])

        temp = self.domains[x].copy()
        for x_word in self.domains[x]:
            if x_word[x_idx] not in possible_overlaps:
                temp.remove(x_word)
                is_revision_made = True
                
        self.domains[x] = temp
                
        return is_revision_made
        raise NotImplementedError

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        all_arcs = [(x,y) for (x,y) in self.crossword.overlaps if self.crossword.overlaps[x,y] is not None]
        if arcs is None:
            arcs = all_arcs
            
        while len(arcs) != 0:
            (x,y) = arcs.pop(0)
            if self.revise(x,y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x):
                    if z is not y:
                        arcs.append((z,x))

        return True
        raise NotImplementedError

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for v in self.crossword.variables:
            if v not in assignment:
                return False
        return True
        raise NotImplementedError

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        assigned_word = set()
        
        for v in assignment:
                
            # check if every value is the correct length
            if len(assignment[v]) != v.length:
                return False
            
            # check if there is conflict between neighboring variables
            for neighbor in self.crossword.neighbors(v):
                if neighbor in assignment:
                    if assignment[v][self.crossword.overlaps[v,neighbor][0]] != assignment[neighbor][self.crossword.overlaps[v,neighbor][1]]:
                        return False

            # check if all values are distinct by comparing the numbers of distinct values and the number of assigned values
            if assignment[v] in assigned_word:
                return False
            
            assigned_word.add(assignment[v])
        
        return True
        raise NotImplementedError

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        
        ruled_outs = dict()
        for v in self.domains[var]:
            ruled_out = 0
            for neighbor in self.crossword.neighbors(var):
                var_idx = self.crossword.overlaps[var,neighbor][0]
                neighbor_idx = self.crossword.overlaps[var,neighbor][1]
                for n in self.domains[neighbor]:
                    if n[neighbor_idx] != v[var_idx]:
                        ruled_out += 1
            ruled_outs[v] = ruled_out

        # sort the dictionary by its value
        ruled_outs = dict(sorted(ruled_outs.items(), key=lambda item: item[1]))

        ordered_domain = []
        for v in ruled_outs:
            ordered_domain.append(v)

        return ordered_domain
        raise NotImplementedError

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        min_domain = math.inf
        selected_var = None
        max_neighbor = - math.inf
        for v in self.crossword.variables:
            if v not in assignment:
                # minimum remaining value heuristic
                if len(self.domains[v]) < min_domain:
                    min_domain = len(self.domains[v])
                    selected_var = v
                # degree heuristic
                if len(self.domains[v]) == min_domain:
                    if len(self.crossword.neighbors(v)) > max_neighbor:
                        max_neighbor = len(self.crossword.neighbors(v))
                        selected_var = v

        return selected_var
        raise NotImplementedError

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
    
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        
        for value in self.order_domain_values(var, assignment):
            temp = assignment.copy()
            temp[var] = value
            
            if self.consistent(temp):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                assignment.pop(var)
                
        return None
        raise NotImplementedError


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
