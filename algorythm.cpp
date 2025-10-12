//
// Created by mikha on 13.10.2025.
//

#include "algorythm.h"
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <limits>
#include <chrono>

// Definicja "nieskończoności" dla kosztów
const int INF = std::numeric_limits<int>::max();

// Typy pomocnicze dla czytelności
using Edge = std::pair<int, int>; // para {sąsiad, waga}
using Graph = std::vector<std::vector<Edge>>; // Graf jako lista sąsiedztwa

// Globalny muteks chroniący dostęp do kolejki priorytetowej
std::mutex pq_mutex;

/**
 * @brief Funkcja zadaniowa dla wątku roboczego.
 * * Przetwarza podzbiór sąsiadów wierzchołka `u`, obliczając nowe koszty dojścia
 * i bezpiecznie aktualizując globalne struktury danych.
 * * @param u Aktualnie przetwarzany wierzchołek.
 * @param neighbors_subset Podzbiór sąsiadów `u` do przetworzenia przez ten wątek.
 * @param dist Referencja do globalnej, atomowej tablicy kosztów.
 * @param pq Referencja do globalnej kolejki priorytetowej.
 */
void relax_edges_task(
        int u,
        const std::vector<Edge>& neighbors_subset,
        std::vector<std::atomic<int>>& dist,
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>>& pq)
{
    for (const auto& edge : neighbors_subset) {
        int v = edge.first;
        int weight = edge.second;

        // Atomowy odczyt aktualnego kosztu dojścia do u
        int dist_u = dist[u].load(std::memory_order_relaxed);

        if (dist_u != INF) {
            int new_dist_v = dist_u + weight;

            // Atomowa aktualizacja kosztu dojścia do v
            // Używamy pętli compare-and-swap (CAS), aby zapewnić poprawność
            int old_dist_v = dist[v].load(std::memory_order_relaxed);
            while (new_dist_v < old_dist_v) {
                // compare_exchange_weak próbuje zamienić old_dist_v na new_dist_v
                // Jeśli się uda, zwraca true. Jeśli nie (bo inny wątek już to zrobił),
                // zwraca false i aktualizuje old_dist_v do nowej wartości z pamięci.
                if (dist[v].compare_exchange_weak(old_dist_v, new_dist_v)) {
                    // Jeśli aktualizacja się powiodła, dodajemy v do kolejki
                    std::lock_guard<std::mutex> lock(pq_mutex);
                    pq.push({new_dist_v, v});
                    break; // Wyjście z pętli CAS
                }
            }
        }
    }
}

/**
 * @brief Główna funkcja wielowątkowego algorytmu Dijkstry.
 * * @param graph Graf wejściowy.
 * @param start_node Wierzchołek startowy.
 * @param num_threads Liczba wątków roboczych do użycia.
 */
void dijkstra_parallel(const Graph& graph, int start_node, int num_threads) {
    int n = graph.size();
    std::vector<std::atomic<int>> dist(n);
    for (int i = 0; i < n; ++i) {
        dist[i].store(INF);
    }
    dist[start_node].store(0);

    // Kolejka priorytetowa przechowuje pary {koszt, wierzchołek}
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    pq.push({0, start_node});

    std::vector<std::thread> threads;

    while (true) {
        int u = -1;

        // Sekcja krytyczna - bezpieczny dostęp do kolejki priorytetowej
        {
            std::lock_guard<std::mutex> lock(pq_mutex);
            if (pq.empty()) {
                break; // Koniec algorytmu, jeśli kolejka jest pusta
            }
            u = pq.top().second;
            pq.pop();
        }

        // Optymalizacja: jeśli znaleźliśmy już lepszą ścieżkę do u, ignorujemy ten wpis
        if (dist[u].load(std::memory_order_relaxed) < pq.top().first && !pq.empty()) {
            continue;
        }

        const auto& neighbors = graph[u];
        if (neighbors.empty()) {
            continue;
        }

        // Dzielimy listę sąsiadów na części (chunks) dla wątków
        int chunk_size = (neighbors.size() + num_threads - 1) / num_threads;
        threads.clear();

        for (int i = 0; i < num_threads; ++i) {
            auto start_it = neighbors.begin() + i * chunk_size;
            if (start_it >= neighbors.end()) break; // Nie twórz wątku, jeśli nie ma dla niego pracy

            auto end_it = neighbors.begin() + std::min((size_t)((i + 1) * chunk_size), neighbors.size());

            std::vector<Edge> subset(start_it, end_it);
            threads.emplace_back(relax_edges_task, u, subset, std::ref(dist), std::ref(pq));
        }

        // Czekamy, aż wszystkie wątki robocze zakończą przetwarzanie sąsiadów wierzchołka u
        for (auto& t : threads) {
            t.join();
        }
    }

    // Wyświetlanie wyników
    std::cout << "Wyniki algorytmu dla " << num_threads << " watkow:\n";
    for (int i = 0; i < n; ++i) {
        int final_dist = dist[i].load();
        std::cout << "Koszt dojscia do " << i << ": " << (final_dist == INF ? "brak sciezki" : std::to_string(final_dist)) << std::endl;
    }
}

int main() {
    // Przykładowy graf do testów
    int num_nodes = 6;
    Graph graph(num_nodes);
    graph[0].assign({{1, 7}, {2, 9}, {5, 14}});
    graph[1].assign({{0, 7}, {2, 10}, {3, 15}});
    graph[2].assign({{0, 9}, {1, 10}, {3, 11}, {5, 2}});
    graph[3].assign({{1, 15}, {2, 11}, {4, 6}});
    graph[4].assign({{3, 6}, {5, 9}});
    graph[5].assign({{0, 14}, {2, 2}, {4, 9}});

    int start_node = 0;
    int num_threads = std::thread::hardware_concurrency(); // Użyj tylu wątków, ile rdzeni ma CPU
    if (num_threads == 0) num_threads = 4; // Zabezpieczenie

    // Pomiar czasu wykonania
    auto start_time = std::chrono::high_resolution_clock::now();

    dijkstra_parallel(graph, start_node, num_threads);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execution_time = end_time - start_time;

    std::cout << "\nCzas wykonania: " << execution_time.count() << " ms\n";

    return 0;
}