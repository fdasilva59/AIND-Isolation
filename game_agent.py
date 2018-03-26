"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    #raise NotImplementedError

    import math

    ## This score is a composite score computed from some of the game metrics

     # Reward a lot if we win
    if game.is_winner(player):
        return float("inf")
    # or penalize a lot if we lost
    elif game.is_loser(player):
        return float("-inf")
    #  or rewards depending of the number of options remaining available
    else: 


        # Prepare metrics to compute a composite score

        # Nb of moves differences between players
        nb_own_possible_moves = len(game.get_legal_moves(player))
        nb_opp_possible_moves = len(game.get_legal_moves(game.get_opponent(player)))
        #sc0 = nb_own_possible_moves - nb_opp_possible_moves

        # Distance from center of board
        own_position = game.get_player_location(player)
        opp_position = game.get_player_location(game.get_opponent(player))
        d = math.sqrt((game.width/2 - own_position[0] ) ** 2 + (game.height/2 - own_position[1]) ** 2)
       
        #print("Debug move nb_own_possible_moves=",nb_own_possible_moves, "nb_opp_possible_moves", nb_opp_possible_moves, "d",d))

        # The final heuristics score consists in the "weighted" difference between the 2 players number of legal moves available
        # Feature scaling is applied to the distance to the center of the board value, which ranges between 0.707 and 4.301 
        score = nb_own_possible_moves - ( 2 * nb_opp_possible_moves * (d-0.707)/(4.301-0.701) )

        return float(score)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    #raise NotImplementedError

    # Reward a lot if we win
    if game.is_winner(player):
        return float("inf")
    # or penalize a lot if we lost
    elif game.is_loser(player):
        return float("-inf")
    #  or rewards depending of the number of options remaining available
    else: 
        # Considering a blank 7x7 board, initialize a static score book containing
        # the number of maximum theoretical legal moves possible at each board positions 
        # using the chess knight moves (L shape)
        # Note : For game board of any size, this should be created by the Board constructor
        score_book = [[2,3,4,4,4,3,2],
                      [3,4,6,6,6,4,3],
                      [4,6,8,8,8,6,4],
                      [4,6,8,8,8,6,4],
                      [4,6,8,8,8,6,4],
                      [3,4,6,6,6,4,3],
                      [2,3,4,4,4,3,2]]

        # Variables to store temporary score for the player (sc) and its opponent (sco)
        sc=0
        sco=0

        # Compute a score for the player, based on the accumulation of the number of 
        # theoretical maximum legal moves possible, for each of the legal moves currently 
        # available to the player
        for m in game.get_legal_moves(player):
            sc = sc + score_book[m[0]][m[1]]
        
        # Same thing, but for the player's opponent
        for m in game.get_legal_moves(game.get_opponent(player)):
            sco = sco + score_book[m[0]][m[1]]

        # The final heuristics score consists in the difference between the 2 players scores
        return float(sc - sco)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.


    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    #raise NotImplementedError

    # Reward a lot if we win
    if game.is_winner(player):
        return float("inf")
    # or penalize a lot if we lost
    elif game.is_loser(player):
        return float("-inf")
    #  or rewards depending of the number of options remaining available
    else:
        
        # Considering a blank 7x7 board, initialize a static score book containing
        # the number of maximum theoretical legal moves possible at each board positions 
        # using the chess knight moves (L shape)
        # Note : For game board of any size, this should be created by the Board constructor
        score_book = [[2,3,4,4,4,3,2],
                      [3,4,6,6,6,4,3],
                      [4,6,8,8,8,6,4],
                      [4,6,8,8,8,6,4],
                      [4,6,8,8,8,6,4],
                      [3,4,6,6,6,4,3],
                      [2,3,4,4,4,3,2]]

        # # previous version (simple accumulation instead of mean score)
        # sc=0
        # sco=0
        # player_moves = game.get_legal_moves(player)
        # opponents_moves = game.get_legal_moves(game.get_opponent(player))

        # for m in player_moves:

        #     fc_game = game.forecast_move(m)
        #     for m2 in fc_game.get_legal_moves(player):
        #         sc = sc + score_book[m2[0]][m2[1]]

        # for m in opponents_moves:

        #     fc_game = game.forecast_move(m)
        #     fc_game._active_player, fc_game._inactive_player = fc_game._inactive_player, fc_game._active_player
        #     for m2 in fc_game.get_legal_moves(player):
        #         sco = sco + score_book[m2[0]][m2[1]]

        # return sc - sco 

        # Variables to store temporary score for the player (sc) and its opponent (sco)
        sc=0
        sco=0

        # Get the legal moves list for each players
        player_moves = game.get_legal_moves(player)
        opponents_moves = game.get_legal_moves(game.get_opponent(player))

        # Compute a mean score for the player, based on the accumulation of the number of 
        # theoretical maximum legal moves possible if the player could play twice in a row
        for m in player_moves:

            sc2=0
            # Explore further the legal moves available if the user could play twice in a row 
            # without taking into account its opponent
            fc_game = game.forecast_move(m)
            for m2 in fc_game.get_legal_moves(player):
                # accumulate the theoritical scores at level 2
                sc2 = sc2 + score_book[m2[0]][m2[1]]

            # compute the theoretical mean at level 2, and accumulate at level 1
            if len (fc_game.get_legal_moves(player))>0 :
                sc = sc + sc2 / len(fc_game.get_legal_moves(player)) 

        # finally, compute the theoretical mean at level 1
        if len(player_moves)>0:
            sc = sc / len(player_moves)


        # Same thing, but for the player's opponent
        for m in opponents_moves:

            sco2 = 0
            fc_game = game.forecast_move(m)
            fc_game._active_player, fc_game._inactive_player = fc_game._inactive_player, fc_game._active_player
            for m2 in fc_game.get_legal_moves(player):
                sco2 = sco2 + score_book[m2[0]][m2[1]]

            if len (fc_game.get_legal_moves(player))>0 :
                sco = sco + sco2 / len (fc_game.get_legal_moves(player)) 

        if len(opponents_moves)>0:
            sco = sco / len(opponents_moves)

        # The final heuristics score consists in the difference between the 2 players mean scores 
        return float(sc - sco)
        



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        #raise NotImplementedError

        def min_value(self, game, depth):
            """ My implementation of MINIMAX algorithm in an helper function to
                facilitate recursion

            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting


            Returns
            -------
            (float)
                Returns the score value

            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # check for terminal state
            legal_moves = game.get_legal_moves()
            if (not legal_moves) or (depth==0):
                return self.score(game, self)

            # MINVALUE
            best_score = float("inf")
            best_score, _ = min([(max_value(self, game=game.forecast_move(m), depth=depth-1), m) for m in legal_moves])
            return best_score


        def max_value(self, game, depth):
            """ My implementation of MINIMAX algorithm in an helper function to
                facilitate recursion

            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting


            Returns
            -------
            (float)
                Returns the score value

            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Check for terminal state
            legal_moves = game.get_legal_moves()
            if (not legal_moves) or (depth==0):
                return self.score(game, self)

            # MAXVALUE
            best_score = float("-inf")
            best_score, _ = max([(min_value(self, game=game.forecast_move(m), depth=depth-1), m) for m in legal_moves])
            return best_score

        # MINIMAX main body 

        # Check for terminal state
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        best_move= legal_moves[0]
        _, best_move = max([(min_value(self, game=game.forecast_move(m), depth=depth-1), m) for m in legal_moves])

        return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        #raise NotImplementedError

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            # Implement Iterative depth search
            iterative_depth=0
            best_move=(-1,-1)
            while(True):
                #print("NEW LEVEL ****************************************** best_move", best_move)
                iterative_depth=iterative_depth+1
                best_move  = self.alphabeta(game, iterative_depth, float("-inf"), float("inf"))

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move




    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        #raise NotImplementedError


        def min_value(self, game, depth, alpha, beta):
            """ My implementation of MINIMAX algorithm in an helper function to
                facilitate recursion

            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting

            alpha : float
                Alpha limits the lower bound of search on minimizing layers

            beta : float
                Beta limits the upper bound of search on maximizing layers

            Returns
            -------
            (float)
                Returns the score value

            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Formated header for Debug info
            #space = self.search_depth - depth
            #ret = "*" * (space)

            #print(ret,"Entering MIN : alphabeta=", alpha,beta)

            # Check terminal state
            legal_moves = game.get_legal_moves()
            if  (not legal_moves) or (depth<=0):
                #print(ret,"MIN Leave termination")
                return self.score(game, self)


            # MINVALUE
            best_score = float("inf")
            #best_score, _ = min([(max_value(self, game=game.forecast_move(m), depth=depth-1), m) for m in legal_moves])

            for m in legal_moves:
                #print(ret, "MIN legal moves", legal_moves, " exploring", m )
                score = max_value(self, game=game.forecast_move(m), depth=depth-1, alpha=alpha, beta=beta)
                #print(ret, "MIN max_value score", score, "previous best_score (choosing the  min)", best_score)
                best_score = min(best_score, score)

                # Alpha beta pruning
                if (best_score <= alpha):
                    #print(ret, "MIN ALPHABETA PRUNING (best_score <= alpha) : not updating beta, returning best score", best_score, "EXIT LOOP")
                    return best_score
                beta = min(beta , best_score)
                #print(ret, "MIN update BETA", beta, "returning best score", best_score)

            #print(ret, "MIN DEBUG currently m", m, "in", legal_moves )
            #print(ret, "MIN LOOP FINISHED : alphabeta=", alpha, beta, "best_score", best_score)
            return best_score


        def max_value(self, game, depth, alpha, beta):
            """ My implementation of MINIMAX algorithm in an helper function to
                facilitate recursion

            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting

            alpha : float
                Alpha limits the lower bound of search on minimizing layers

            beta : float
                Beta limits the upper bound of search on maximizing layers


            Returns
            -------
            (float)
                Returns the score value

            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Formated header for Debug info
            #space = self.search_depth - depth
            #ret = "*" * (space)

            #print(ret, "Entering MAX : alphabeta=", alpha, beta)

            legal_moves = game.get_legal_moves()
            if  (not legal_moves) or (depth<=0):
                #print(ret, "MAX Leave termination")
                return self.score(game, self)

            # MAXVALUE
            best_score = float("-inf")

            for m in legal_moves:
                #print(ret, "MAX legal moves", legal_moves, " exploring", m )
                score= min_value(self, game=game.forecast_move(m), depth=depth-1, alpha=alpha, beta=beta)
                #print(ret, "MAX min_value score", score, "previous best_score (choosing the  max)", best_score)
                best_score = max(best_score, score)

                # Alpha beta pruning
                if (best_score >= beta):
                    #print(ret, "MAX ALPHABETA PRUNING (best_score >= beta) : not updating alpha, returning best score", best_score, "EXIT LOOP")
                    return best_score
                alpha = max(alpha, best_score)
                #print(ret, "MAX update ALPHA",alpha, "returning best score", best_score)

            #print(ret, "MAX DEBUG currently m", m, "in", legal_moves )
            #print(ret, "MAX LOOP FINISHED : alphabeta=", alpha, beta, "best_score", best_score)
            return best_score

        # Main body Alpha Beta Full Search

        # Check if not already in terminal state
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        best_score = float("-inf")
        best_move= legal_moves[0]  

        #i=1
        for m in legal_moves:
            #print("\n\nROOT (", i,"/",len(legal_moves), ") legal moves", legal_moves, " exploring", m )
            score= min_value(self, game=game.forecast_move(m), depth=depth-1, alpha=alpha,beta=beta)
            #i = i+1
            #print("--ROOT score", score, "at pos", m, "while current best_move was ", best_move, "aplhabeta", alpha,beta)
            if (score > best_score):
                best_score = score
                #alpha = best_score
                best_move = m
                #print("--ROOT UPDATE BEST score", best_score, "best_move", best_move, "aplhabeta", alpha,beta)
            alpha=max(alpha, best_score)

        return best_move
