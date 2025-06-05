#include <iostream>
#include <thread>
#include <queue>
#include <random>
#include <semaphore>
#include <chrono>
#include <vector>
#include <mutex>
using namespace std;

struct Task {
	int id;
	int priority;
	int time;
	bool operator<(const Task& other) const {
		return priority > other.priority; 
	}
};

priority_queue<Task> task_queue;
counting_semaphore<3> servers(3);
mutex output_mutex;
random_device rd;
mt19937 gen(rd());

void add_task(int id, int priority) {
	uniform_int_distribution<> dis(1, 5);
		Task task = { id, priority, dis(gen) };
	task_queue.push(task);
	lock_guard<mutex> guard(output_mutex);
	cout << "Task " << id << " with priority " << priority << " added to the queue.\n";
}

void process_task(Task task, int thread_id) {
	servers.acquire();
	this_thread::sleep_for(chrono::seconds(task.time));
	{
		lock_guard<mutex> guard(output_mutex);
		cout << "Task " << task.id << " (priority " << task.priority << ") completed after "
		<< task.time << " seconds. Server: " << thread_id + 1 << endl;
	}
	servers.release();
}

void process_tasks(int thread_id) {
	while (!task_queue.empty()) {
		Task task = task_queue.top();
		task_queue.pop();
		process_task(task, thread_id);
	}
}

int main() {
	for (int i = 0; i < 10; ++i) {
		uniform_int_distribution<> dis(1, 5);
		add_task(i+1, dis(gen));
	}

	vector<thread> threads;
	for (int i = 0; i < 5; ++i) {
		threads.push_back(thread(process_tasks, i));
	}
	for (auto& t : threads) {
		t.join();
	}
	return 0;
}