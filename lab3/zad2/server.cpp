#include <boost/asio.hpp>
#include <iostream>
#include <sstream>
#include <memory>
using boost::asio::ip::tcp;
using namespace std;

class Session : public enable_shared_from_this<Session> {
public:
	Session(tcp::socket socket) : socket_(move(socket)) {}
	void start() {
		do_read();
	}
private:
	void do_read() {
		auto self(shared_from_this());
		socket_.async_read_some(boost::asio::buffer(data_),
			[this, self](boost::system::error_code ec, size_t length) {
				if (!ec) {
					do_write(); // Эхо-ответ
				}
			});
	}
	void do_write() {
		auto self(shared_from_this());
		istringstream iss(data_);
		int num;
		int max_num = -32000;
		while (iss >> num) {
			if (num > max_num) max_num = num;
		}
		string mn = to_string(max_num);
		boost::asio::async_write(socket_, boost::asio::buffer(mn, size(mn)),
			[this, self](boost::system::error_code ec, size_t length) {
				if (!ec) {
					do_read(); // Продолжаем читать
				}
			});
	}
	tcp::socket socket_;
	char data_[1024];
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
