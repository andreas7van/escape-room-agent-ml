class Puzzle:
    """
    Αναπαριστά έναν γρίφο μέσα στο Escape Room.

    Για το reinforcement learning, ο agent δεν γράφει κείμενο ως απάντηση.
    Η επιτυχία θεωρείται ότι γίνεται όταν ο agent βρίσκεται στο σωστό κελί
    και εκτελεί την ενέργεια SOLVE_PUZZLE.

    Αυτό επιτρέπει στον agent να μάθει:
    - σε ποιο κελί βρίσκεται ο γρίφος
    - πότε πρέπει να εκτελέσει την ενέργεια επίλυσης
    - ότι η επίλυση γρίφου οδηγεί σε θετική ανταμοιβή
    """

    def __init__(self, puzzle_id, symbol, question, answer, reward, position):
        self.puzzle_id = puzzle_id
        self.symbol = symbol
        self.question = question
        self.answer = answer
        self.reward = reward
        self.position = tuple(position)
        self.solved = False

    def solve(self):
        if self.solved:
            return False, -5, "Puzzle already solved."

        self.solved = True
        return True, self.reward, f"Puzzle solved: {self.puzzle_id}"

    def reset(self):
        self.solved = False
