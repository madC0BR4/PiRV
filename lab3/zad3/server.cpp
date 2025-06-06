	#include <boost/asio.hpp>
	#include <iostream>
	#include <sstream>
	#include <string>
	#include <vector>
	#include <chrono>
	#include <memory>
	using boost::asio::ip::tcp;
	using namespace std;

	class Session : public enable_shared_from_this<Session> {
	public:
		Session(tcp::socket socket) : socket_(move(socket)), timer_(socket_.get_executor()) {}
		void start() {
			do_read();
		}
	private:
		void do_read() {
			auto self(shared_from_this());
			socket_.async_read_some(boost::asio::buffer(data_, max_length),
				[this, self](boost::system::error_code ec, size_t length) {
					if (!ec) {
						string received(data_, length);
						istringstream iss(received);
						string s;
						vector<string> v;
						while (iss >> s) {
							v.push_back(s);
						}
						if (v.empty()) {
							do_read(); // Продолжаем чтение, если нет данных
							return;
						}
						if (v[0] == "timer") {
							int seconds = stoi(v[1]);
							timer_function(seconds);
						}
					}
				});
		}
		void timer_function(int seconds) {
			auto self(shared_from_this());
			string msg = "Ready in " + to_string(seconds) + " sec\n";
			boost::asio::async_write(socket_, boost::asio::buffer(msg),
				[this, self, seconds](boost::system::error_code ec, size_t) {
					if (!ec) {
						timer_.expires_after(boost::asio::chrono::seconds(seconds));
						timer_.async_wait(
							[this, self](const boost::system::error_code& ec) {
								if (ec) {
									std::cerr << "Timer error: " << ec.message() << "\n";
									return;
								}
								string done = "Done!\n";
								boost::asio::async_write(socket_, boost::asio::buffer(done),
									[](boost::system::error_code, size_t) {});
							});
					}
				});
		}
		tcp::socket socket_;
		boost::asio::steady_timer timer_;
		enum { max_length = 1024 };
		char data_[max_length] = {0};
	};
	class Server {
	public:
		Server(boost::asio::io_context& io_context, short port)
			: acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
			do_accept();
		}
	private:
		void do_accept() {
			acceptor_.async_accept(
				[this](boost::system::error_code ec, tcp::socket socket) {
					if (!ec) {
						make_shared<Session>(move(socket))->start();
					}
					do_accept();
				});
		}
		tcp::acceptor acceptor_;
	};
	int main() {
		try {
			boost::asio::io_context io;
			Server s(io, 12345);
			cout << "Run server 127.0.0.1:12345\n";
			io.run();
		}
		catch (exception& e) {
			cerr << "Error: " << e.what() << endl;
		}
	}
