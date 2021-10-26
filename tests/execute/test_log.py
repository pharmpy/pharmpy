import pharmpy.workflows.log


def test_log():
    client = pharmpy.workflows.log.Log()
    client.log_error("help!")
    client.log_warning("an annoying warning")
    df = client.to_dataframe()
    assert list(df['category']) == ['ERROR', 'WARNING']
    assert list(df['message']) == ['help!', 'an annoying warning']
