#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>

const int POPULATION_SIZE = 20;
const int ELITISM_COUNT = 2;
const double CROSSOVER_RATE = 0.7;
const double MUTATION_RATE = 0.1;

struct Individual {
    std::vector<double> genes;
    double score; // Z-value (fitness)
};

double randomDouble(double min, double max) {
    return min + static_cast<double>(rand()) / RAND_MAX * (max - min);
}


// скрещивание
Individual crossover(const Individual& parent1, const Individual& parent2) {
    Individual child = parent1;
    int num_genes = parent1.genes.size();
    
    if (num_genes > 1 && randomDouble(0, 1) < CROSSOVER_RATE) {
        int p1 = 1 + rand() % (num_genes - 1); // точка разрыва
        
        for (int i = p1; i < num_genes; ++i) {
            child.genes[i] = parent2.genes[i];
        }
    }
    return child;
}

void mutate(Individual& ind, const std::vector<double>& min_vals, const std::vector<double>& max_vals) {
    for (size_t i = 0; i < ind.genes.size(); ++i) {
        if (randomDouble(0, 1) < MUTATION_RATE) {
            ind.genes[i] = randomDouble(min_vals[i], max_vals[i]);
        }
    }
}

const Individual& selectParent(const std::vector<Individual>& pop) {
    int idx1 = rand() % pop.size();
    int idx2 = rand() % pop.size();
    if (pop[idx1].score > pop[idx2].score) return pop[idx1];
    return pop[idx2];
}


extern "C" {
    struct GeneticContext {
        std::vector<Individual> population;
        std::vector<double> min_vals;
        std::vector<double> max_vals;
        int num_params;
        int current_generation;
    };

    GeneticContext* create_optimizer(int num_params) {
        srand(time(0));
        auto* ctx = new GeneticContext;
        ctx->num_params = num_params;
        ctx->current_generation = 0;
        std::cout << "[C++] Генетический оптимизатор создан." << std::endl;
        return ctx;
    }

    void destroy_optimizer(GeneticContext* ctx) {
        if (ctx) delete ctx;
    }

    void init_population(GeneticContext* ctx, double* min_vals, double* max_vals) {
        if (!ctx) return;
        
        ctx->min_vals.assign(min_vals, min_vals + ctx->num_params);
        ctx->max_vals.assign(max_vals, max_vals + ctx->num_params);
        
        ctx->population.clear();
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            Individual ind;
            ind.score = -1e9; // Пока не оценена
            for (int j = 0; j < ctx->num_params; ++j) {
                ind.genes.push_back(randomDouble(ctx->min_vals[j], ctx->max_vals[j]));
            }
            ctx->population.push_back(ind);
        }
        std::cout << "[C++] Популяция инициализирована." << std::endl;
    }

    // получить параметры особи i
    void get_individual(GeneticContext* ctx, int index, double* out_genes) {
        if (ctx && index >= 0 && index < ctx->population.size()) {
            const auto& genes = ctx->population[index].genes;
            std::copy(genes.begin(), genes.end(), out_genes);
        }
    }

    void set_score(GeneticContext* ctx, int index, double score) {
        if (ctx && index >= 0 && index < ctx->population.size()) {
            ctx->population[index].score = score;
        }
    }

    // шаг эволюции
    void evolve(GeneticContext* ctx) {
        if (!ctx) return;

        std::sort(ctx->population.begin(), ctx->population.end(), 
            [](const Individual& a, const Individual& b) {
                return a.score > b.score; 
            });

        std::cout << "[C++] Поколение " << ctx->current_generation 
                  << ", Лучший Score: " << ctx->population[0].score << std::endl;

        std::vector<Individual> new_pop;

        for (int i = 0; i < ELITISM_COUNT; ++i) {
            new_pop.push_back(ctx->population[i]);
        }

        while (new_pop.size() < POPULATION_SIZE) {
            const Individual& p1 = selectParent(ctx->population);
            const Individual& p2 = selectParent(ctx->population);
            
            Individual child = crossover(p1, p2);
            mutate(child, ctx->min_vals, ctx->max_vals);
            
            new_pop.push_back(child);
        }
        
        ctx->population = new_pop;
        ctx->current_generation++;
    }
}
