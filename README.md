# Multiplayer Tic Tac Toe

This is a multiplayer implementation of the classic Tic Tac Toe game using Python, Pygame, and socket programming for networking.

## Running the Game

To run the multiplayer Tic Tac Toe game, you need to follow these steps:

1. **Start the Server**:
   - Open a terminal or command prompt.
   - Navigate to the directory containing the `server.py` file.
   - Run the server script with the following command:

     python3 server.py

   - The server will start running and listen for incoming client connections.

2. **Start the Clients**:
   - Open a new terminal or command prompt window for each client you want to run.
   - Navigate to the directory containing the `client.py` file and the `assets` folder with the game images.
   - Run the client script with the following command:

     python3 client.py

   - Repeat this step in a new terminal or command prompt window for the second client.

3. **Play the Game**:
   - After running the server and two client instances, you should see the Tic Tac Toe game board on each client window.
   - The first client that connects will be assigned the 'X' symbol, and the second client will be assigned the 'O' symbol.
   - The game will start with the 'X' player's turn.
   - Players can make their moves by clicking on the desired cell in their respective client windows.
   - After each move, the updated board state will be visible on both clients.
   - The game will continue until a player wins or it's a draw.
   - If a player wins or the game ends in a draw, both clients will display the result, and the board will be reset for a new game.

4. **Exit the Game**:
   - To exit the game, simply close the terminal or command prompt windows for the server and clients.

Note: Make sure you have the required dependencies installed, such as the Python socket module and the Pygame library.
