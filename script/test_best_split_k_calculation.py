#!/usr/bin/env python3

import numpy as np

def find_closest_k(Ksplit, grid_size, N):  
    # Find the minimum value of k such that [(k+Ksplit)*grid_size] mod N is minimized  
    #min_remainder = float('inf')
    min_cost = float('inf')
    best_k = 0  
      
    for k in range(-Ksplit+1, N-Ksplit):  
        remainder = ((Ksplit + k) * grid_size) % N
        wave_quant_cost = (N-remainder)*(N-remainder)
        relative_k_change = 100.0 * (k/Ksplit)
        k_change_cost = relative_k_change * relative_k_change
        cost = k_change_cost + 2*wave_quant_cost
        #print(f"k = {k}, N-remainder = {N-remainder}, k_change_cost = {k_change_cost}, (N-remainder)^2 = {wave_quant_cost}, cost = {cost}")
        if cost < min_cost:  
            min_cost = cost  
            best_k = k 
        elif cost == min_cost:
            # For equal candidates,choose the one with the smallest absolute value
            if abs(k) < abs(best_k):
              best_k = k
            elif abs(k) == abs(best_k):
              # If both have the same absolute value, choose the one with the larger k
              best_k = max(best_k, k)
              
    return best_k  

def find_k(K, grid_size, N):  
    if (K * grid_size) % N == 0:
        return 0
    return find_closest_k(K, grid_size, N)

def test(K,G,N):
  k = find_k(K, G, N)
  print(f"Find K0 such that [(K0+{K})*{G}] mod {N} is minimized.")
  print(f"The value for K={K}, grid_size={G}, N={N} is: K0 = {k}")
  print(f"We have K = {K} -> K + K0 = {K + k} and [(K0 + K)*grid_size]mod N = {((k+K)*G)%N} vs [K x grid_size] mod N = {(K*G)%N} \n")
  
# Main function to test the calculation of K
def main():
    N = 304

    max_occupancy = 12
    grid_size = 4 * 4
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)

    max_occupancy = 15
    grid_size = 8 * 8
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)

    max_occupancy = 5
    grid_size = 16 * 16
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)

    max_occupancy = 10
    grid_size = 16 * 16
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)

    max_occupancy = 7
    grid_size = 16 * 32
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)

    max_occupancy = 9
    grid_size = 25
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)

    max_occupancy = 5
    grid_size = 30
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)
    
    max_occupancy = 1
    grid_size = 25
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)

    max_occupancy = 7
    grid_size = 288
    K = int(round((max_occupancy * N) / grid_size))
    test(K, grid_size, N)

if __name__ == "__main__":
    main()