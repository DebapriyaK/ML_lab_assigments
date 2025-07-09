def range(a):
    m=min(a)
    M=max(a)
    ran=M-m
    
    return ran


def main():
    arr=input("enter list")
    arr = list(map(int, arr.split()))
    if(len(arr)<3):
        print("Range determination not possible")
    else:
        print(range(arr))
        
if __name__ == "__main__":
    main()
