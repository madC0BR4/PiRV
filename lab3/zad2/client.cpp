#include <boost/asio.hpp>
#include <iostream>
using boost::asio::ip::tcp;
using namespace std;

int main() {
	try {
		boost::asio::io_context io;
		tcp::resolver resolver(io);
		auto endpoints = resolver.resolve("127.0.0.1", "12345");
		tcp::socket socket(io);
		boost::asio::connect(socket, endpoints);
		string msg;
		cout << "Enter numbers: ";
		getline(cin, msg);
		boost::asio::write(socket, boost::asio::buffer(msg));
		char reply[1024];
		size_t len = socket.read_some(boost::asio::buffer(reply));
		cout << "Maximum number is: " << string(reply, len) << "\n";
	}
	catch (exception& e) {
		cerr << "Error: " << e.what() << "\n";
	}
}