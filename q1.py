def count_pairs_with_sum(a):
    count = 0
    n = len(a)

    for i in range(n):
        for j in range(i + 1, n):
            if a[i]+a[j]==10:
                count+=1

    return count


def main():
    arr=[2, 7, 4, 1, 3, 6]
    count=count_pairs_with_sum(arr)
    print("Number of pairs with sum 10:", count)


if __name__ == "__main__":
    main()
