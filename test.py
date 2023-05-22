def compute_closest_to_zero(ts):
    # Write your code here
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    if ts is None or len(ts) ==  0:
      return 0

    max_val = float('inf')
    for e  in ts:
        if abs(e) < abs(max_val):
            max_val = e
        elif abs(max_val) == abs(e):
            max_val = abs(e)
            
    return max_val      

print(compute_closest_to_zero([45, -98, 22, 75, -1, 84]))

print(compute_closest_to_zero([]))

print(compute_closest_to_zero([45, 7, -98, 22, 75, -7, 84]))
