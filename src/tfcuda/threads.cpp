//
// Created by connor on 2/22/22.
//

#include "thread"

void test(long i, long j){
    int k = 0;
    for (k = 0; k < i * j; k++){
        if (k % 100000000 == 0)
            printf("hello from %d\n", j);
    }
    printf("hello %d\n", k);
}

int main(int argc, char **argv) {
    std::thread th1(test, 100000, 21000);

    std::thread th2(test, 100000, 21001);

    th1.join();
    th2.join();

}
