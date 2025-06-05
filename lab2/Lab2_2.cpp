#include <iostream>
#include <thread>
#include <queue>
#include <random>
#include <semaphore>
#include <chrono>
#include <vector>
#include <mutex>
using namespace std;

mutex mtx;

struct CrossRoad {
	int id;
	int green_time_1;
	int green_time_2;
	bool side;
};

struct TrafficQueue {
	int id;
	int queue_time;
	bool emergency_flag;
};

static const int GREEN_LIGHT_STANDART = 20;
static const int SIMULATING_TIME = 100;
vector<CrossRoad> CrossVector(10);
vector<TrafficQueue> TrafficVector(20);

int genNum(int a, int b) {
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	uniform_int_distribution<int> distribution(a, b);
	int random_number = distribution(generator);
	return random_number;
}

void CreateCrossRoads() {
	for (int i = 0; i < 10; i++) {
		CrossRoad cs = { i, GREEN_LIGHT_STANDART, GREEN_LIGHT_STANDART, false };
		CrossVector[i] = cs;

		int time = genNum(0, 1);
		bool flag = false;
		if (genNum(1, 10) == 3) {
			flag = true;
		}
		TrafficQueue tq = { i, time, flag };
		TrafficVector[i] = tq;

		time = genNum(0, 1);
		flag = false;
		if (genNum(1, 10) == 3) {
			flag = true;
		}
		tq = { i + 10, time, flag };
		TrafficVector[i + 10] = tq;
	}
}

void TrafficGeneration() {
	for (int i = 0; i < 20; i++) {
		int time = 0;
		if (genNum(1, 10) == 3) {
			time = 1;
		}
		bool flag = false;
		if (genNum(1, 10) == 3) {
			flag = true;
		}
		TrafficVector[i].queue_time += time;
		TrafficVector[i].emergency_flag = flag;
	}
}

void ChangeCrossTime(int i) {
	if (CrossVector[i].side == false) {
		//int all_traffic = TrafficVector[i].queue_time + TrafficVector[i + 10].queue_time;
		if ((TrafficVector[i].queue_time * 100 / CrossVector[i].green_time_1) > 70 ) {
			CrossVector[i].green_time_1 *= 1.5;
		}
		if ((TrafficVector[i].queue_time * 100 / CrossVector[i].green_time_1) < 50) {
			CrossVector[i].green_time_1 = GREEN_LIGHT_STANDART;
		}
	}
	if (CrossVector[i].side == true) {
		//int all_traffic = TrafficVector[i].queue_time + TrafficVector[i + 10].queue_time;
		if ((TrafficVector[i+10].queue_time * 100 / CrossVector[i].green_time_2) > 70) {
			CrossVector[i].green_time_2 *= 1.5;
		}
		if ((TrafficVector[i+10].queue_time * 100 / CrossVector[i].green_time_2) < 50) {
			CrossVector[i].green_time_2 = GREEN_LIGHT_STANDART;
		}
	}
}

void CheckTraffic(int i) {
	if (CrossVector[i].side == false) {
		TrafficVector[i].queue_time -= 1;
		if (TrafficVector[i].queue_time < 0) {
			TrafficVector[i].queue_time = 0;
		}
	}
	if (CrossVector[i].side == true) {
		TrafficVector[i + 10].queue_time -= 1;
		if (TrafficVector[i + 10].queue_time < 0) {
			TrafficVector[i + 10].queue_time = 0;
		}
	}
}

void CrossSimulating(int i) {
	int change_flag_time = CrossVector[i].green_time_1;
	for (int j = 0; j < SIMULATING_TIME; j++) {
		TrafficGeneration();
		if (change_flag_time == 0) {
			ChangeCrossTime(i);
			if (CrossVector[i].side == false) {
				CrossVector[i].side = true;
				change_flag_time = CrossVector[i].green_time_2;
			}
			if (CrossVector[i].side == true) {
				CrossVector[i].side = false;
				change_flag_time = CrossVector[i].green_time_1;
			}
		}
		change_flag_time -= 1;
		CheckTraffic(i);
		if (j % 5 == 0) {
			mtx.lock();
			cout << "Cross: " << CrossVector[i].id << ". Traffic: " << (TrafficVector[i].queue_time * 100 / CrossVector[i].green_time_1) << "%." << endl;
			mtx.unlock();
		}
	}
}

int main() {
	CreateCrossRoads(); 
	vector<thread> threads;

	for (int i = 0; i < 10; ++i) {
		threads.push_back(thread(CrossSimulating, i));
	}
	for (auto& t : threads) {
		t.join();
	}

	/*for (int j = 0; j < 5; j++) {
		TrafficGeneration();
		for (int i = 0; i < 20; i++) {
			cout << "Cross: " << TrafficVector[i].id << ", time: " << TrafficVector[i].queue_time << ", emergency - " << TrafficVector[i].emergency_flag << endl;
		}
		cout << "----------------------------------------------------" << endl;
	}*/

	return 0;
}