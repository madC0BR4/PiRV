
#include <boost/asio.hpp>
#include <iostream>

using boost::asio::ip::tcp;
using namespace std;

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), 12345));

        while (true) {
            cout << "Enter number to calculate factorial: ";
            string input;
            getline(std::cin, input);
            if (input.empty()) break;

            boost::asio::write(socket, boost::asio::buffer(input + "\n"));

            boost::asio::streambuf response;
            boost::asio::read_until(socket, response, '\n');

            istream is(&response);
            string line;
            getline(is, line);
            cout << "Answer from server: " << line << "\n";
        }
    } catch (exception& e) {
        cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}
