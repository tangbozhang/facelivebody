#ifndef SIMPLE_CRYPTO_H_
#define SIMPLE_CRYPTO_H_

#include <string>

void request_receipts(const char *filename, std::string &req_orig);
void generate_receipts(const char *filename);
void verify_receipts(const char *filename, std::string &ver_orig);
std::string get_output_filename(const std::string &filename);

#endif  // SIMPLE_CRYPTO_H_
