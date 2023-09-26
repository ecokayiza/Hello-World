#include <iostream>
#include <stack>
#include <vector>

using namespace std;

const int ROW = 20;
const int COL = 20;

// Define the start and goal positions
const pair<int, int> start_pos = {0, 0};
const pair<int, int> goal_pos = {rand(20), (20)};

// Define the obstacles
const vector<pair<int, int>> obstacles = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};

// Define the grid
int grid[ROW][COL];

// Define the moves
const vector<pair<int, int>> moves = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};

// Check if a move is valid
bool is_valid_move(int row, int col) {
    if (row < 0 || row >= ROW || col < 0 || col >= COL) {
        return false;
    }
    for (auto obstacle : obstacles) {
        if (row == obstacle.first && col == obstacle.second) {
            return false;
        }
    }
    return true;
}

// Perform DFS
vector<pair<int, int>> dfs(pair<int, int> start, pair<int, int> goal) {
    stack<pair<int, int>> s;
    vector<pair<int, int>> path;
    s.push(start);
    while (!s.empty()) {
        auto curr = s.top();
        s.pop();
        if (curr == goal) {
            path.push_back(curr);
            return path;
        }
        if (grid[curr.first][curr.second] == 0) {
            grid[curr.first][curr.second] = 1;
            for (auto move : moves) {
                int row = curr.first + move.first;
                int col = curr.second + move.second;
                if (is_valid_move(row, col)) {
                    s.push({row, col});
                }
            }
        }
    }
    return path;
}

int main() {
    // Initialize the grid
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            grid[i][j] = 0;
        }
    }
    // Mark the obstacles
    for (auto obstacle : obstacles) {
        grid[obstacle.first][obstacle.second] = -1;
    }
    // Perform DFS
    auto path = dfs(start_pos, goal_pos);
    // Print the path
    if (path.empty()) {
        cout << "No path found" << endl;
    } else {
        for (auto pos : path) {
            cout << "(" << pos.first << ", " << pos.second << ")" << endl;
        }
    }
    return 0;
}
