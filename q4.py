def occurrence(a):
    
    n = len(a)
    for i in range(n):
            count=1
            for j in range(i+1,n):
                if a[i]==a[j]:
                    count += 1
            print("Occurrence of"+a[i]+"is",count)
            
def main():
    inp_word = input("Enter a word: ")
    word = list(inp_word)
    occurrence(word)


if __name__ == "__main__":
    main()
