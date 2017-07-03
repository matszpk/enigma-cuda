#include "runner.h"

int main(int argc, char *argv[])
try
{
    Runner runner;
    bool ok;

    ok = runner.settings.FromCommandLine(argc, argv);
    if (!ok)
    {
        CleanUpGPU();
        return 1;
    }

    ok = runner.Initialize();
    if (!ok)
    {
        CleanUpGPU();
        return 2;
    }

    ok = runner.Run();
    if (!ok)
    {
        CleanUpGPU();
        return 3;
    }

    CleanUpGPU();
    return 0;
}
catch(const std::exception& ex)
{
    std::cerr << "error: " << ex.what() << std::endl;
    CleanUpGPU();
    return 1;
}
catch(...)
{
    CleanUpGPU();
    throw;
}
