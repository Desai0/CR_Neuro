#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

using Point = std::vector<double>;

double euclidean_distance_sq(const Point& a, const Point& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum; // квадрат расстояния
}

std::pair<std::vector<int>, std::vector<Point>> k_means_simple(const std::vector<Point>& data, int k, int max_iterations) {
    if (data.empty() || k <= 0) return {};

    int n_points = data.size();
    int n_dims = data[0].size();

    std::vector<Point> centroids;
    if (n_points >= k) {
        for(int i=0; i<k; ++i) centroids.push_back(data[i]);
    } else {
        centroids = data; // eсли меньше чем K
        k = n_points;
    }

    std::vector<int> clusters(n_points);

    for (int iter = 0; iter < max_iterations; ++iter) {
        // точки кластерам
        for (int i = 0; i < n_points; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            for (int j = 0; j < k; ++j) {
                double dist = euclidean_distance_sq(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            clusters[i] = best_cluster;
        }

        // пересчет
        std::vector<Point> new_centroids(k, Point(n_dims, 0.0));
        std::vector<int> counts(k, 0);

        for (int i = 0; i < n_points; ++i) {
            int cluster_id = clusters[i];
            for (int d = 0; d < n_dims; ++d) {
                new_centroids[cluster_id][d] += data[i][d];
            }
            counts[cluster_id]++;
        }

        bool converged = true;
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (int d = 0; d < n_dims; ++d) {
                    new_centroids[j][d] /= counts[j];
                }
            }
            if (euclidean_distance_sq(new_centroids[j], centroids[j]) > 1e-4) {
                converged = false;
            }
        }

        centroids = new_centroids;
        if (converged) break;
    }

    return {clusters, centroids};
}



extern "C" {

    struct SimpleKMeansContext {
        int k;
        int max_iter;
    };

    // контекст
    SimpleKMeansContext* create_kmeans(int k, int max_iter) {
        SimpleKMeansContext* ctx = new SimpleKMeansContext;
        ctx->k = k;
        ctx->max_iter = max_iter;
        std::cout << "[C++] Контекст K-Means создан: K=" << k << std::endl;
        return ctx;
    }

    void destroy_kmeans(SimpleKMeansContext* ctx) {
        if (ctx) delete ctx;
    }

    // Главная функция: найти лучший кластер
    void calculate_best_position(SimpleKMeansContext* ctx, int* x_arr, int* y_arr, int n, int* out_x, int* out_y, int* out_size) {
        if (!ctx || n <= 0) return;

        std::vector<Point> data(n);
        for (int i = 0; i < n; ++i) {
            data[i] = {static_cast<double>(x_arr[i]), static_cast<double>(y_arr[i])};
        }

        auto result = k_means_simple(data, ctx->k, ctx->max_iter);
        std::vector<int>& clusters = result.first;
        std::vector<Point>& centroids = result.second;

        // самый плотный
        std::vector<int> counts(centroids.size(), 0);
        for (int c : clusters) {
            if (c >= 0 && c < counts.size()) counts[c]++;
        }

        int best_cluster_idx = -1;
        int max_points = -1;

        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > max_points) {
                max_points = counts[i];
                best_cluster_idx = i;
            }
        }

        if (best_cluster_idx != -1) {
            *out_x = static_cast<int>(centroids[best_cluster_idx][0]);
            *out_y = static_cast<int>(centroids[best_cluster_idx][1]);
            *out_size = max_points;
        } else {
            *out_x = x_arr[0];
            *out_y = y_arr[0];
            *out_size = 1;
        }
    }
}



// --- Задача о Рюкзаке ---

int knapsack_rec(int w, int i, std::vector<std::vector<int>>& memo, const std::vector<int>& weights, const std::vector<int>& values) {
    if (i == 0 || w == 0) {
        return 0;
    }
    if (memo[i][w] != -1) {
        return memo[i][w];
    }
    if (weights[i - 1] <= w) {
        memo[i][w] = std::max(
            values[i - 1] + knapsack_rec(w - weights[i - 1], i - 1, memo, weights, values), 
            knapsack_rec(w, i - 1, memo, weights, values)
        );
    }
    else {
        memo[i][w] = knapsack_rec(w, i - 1, memo, weights, values);
    }
    return memo[i][w];
}

// какие предметы взяли
void traceback_knapsack(int w, int i, const std::vector<std::vector<int>>& memo, const std::vector<int>& weights, std::vector<int>& selected_indices) {
    if (i == 0 || w == 0) return;
    
    int val_skip = memo[i-1][w]; // если не брать
    if (memo[i][w] != val_skip) {
        selected_indices.push_back(i - 1);
        traceback_knapsack(w - weights[i-1], i - 1, memo, weights, selected_indices);
    } else {
        traceback_knapsack(w, i - 1, memo, weights, selected_indices);
    }
}


extern "C" {
    int solve_knapsack(int* weights, int* values, int n, int capacity, int* out_indices, int* out_count) {
        if (n <= 0 || capacity <= 0) {
            *out_count = 0;
            return 0;
        }

        std::vector<int> v_weights(weights, weights + n);
        std::vector<int> v_values(values, values + n);
        std::vector<std::vector<int>> memo(n + 1, std::vector<int>(capacity + 1, -1));

        int max_val = knapsack_rec(capacity, n, memo, v_weights, v_values);

        std::vector<int> selected;
        traceback_knapsack(capacity, n, memo, v_weights, selected);

        // Копируем результат
        *out_count = selected.size();
        for(size_t i=0; i<selected.size(); ++i) {
            out_indices[i] = selected[i];
        }

        return max_val;
    }
}
