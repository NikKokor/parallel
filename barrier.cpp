#include "barrier.h"

barrier::barrier(unsigned T) : Tmax(T) {
    this->T = T;
};

void barrier::arrive_and_wait(){
    unique_lock lock(mtx);
    if(--T == 0){
        lock_oddity = !lock_oddity;
        T = Tmax;
        cv.notify_all();
    } else {
        auto my_lock = lock_oddity;
        while (my_lock == lock_oddity)
            cv.wait(lock);
    }
};