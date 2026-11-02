namespace Lemolatoon.Udoncl
{
    public enum UdonClStatus
    {
        Success = 0,
        NotInitialized = -1,
        Busy = -2,
        InvalidArgument = -3,
        AbiMismatch = -4,
        ExecutionFailure = -5,
        Disposed = -6,
    }
}
