import pytest
import ROOT
import ROOT._distrdf


class TestRunGraphs:
    """Tests usage of RunGraphs function with Dask backend"""

    @pytest.mark.parametrize("datasource", ["ttree", "rntuple"])
    def test_rungraphs_dask_3histos(self, payload, datasource):
        """
        Submit three different Dask RDF graphs concurrently
        """
        # Create a test file for processing
        treename = "tree"
        filename = f"../data/{datasource}/distrdf_roottest_check_rungraphs.root"
        nentries = 10000
        connection, _ = payload
        df = ROOT.RDataFrame(treename, filename, executor=connection, npartitions=2)
        histoproxies = [
            df.Histo1D((col, col, 1, 40, 45), col)
            for col in ["b1", "b2", "b3"]
        ]

        # Before triggering the computation graphs values are None
        for proxy in histoproxies:
            assert proxy.proxied_node.value is None

        ROOT.RDF.RunGraphs(histoproxies)

        # After RunGraphs all histograms are correctly assigned to the
        # node objects
        for proxy in histoproxies:
            histo = proxy.proxied_node.value
            assert isinstance(histo, ROOT.TH1D)
            assert histo.GetEntries() == nentries
            assert histo.GetMean() == 42

    @pytest.mark.parametrize("datasource", ["ttree", "rntuple"])
    def test_rungraphs_dask_no_premature_return(self, payload, datasource):
        """
        Submit three independent Dask RDF graphs concurrently and check that
        RunGraphs does not return until all of them have been fully processed.
        """
        # Create a test file for processing
        treename = "tree"
        filename = f"../data/{datasource}/distrdf_roottest_check_rungraphs.root"
        nentries = 10000
        connection, _ = payload
        # Three independent dataframes, i.e. three independent computation graphs
        histoproxies = [
            ROOT.RDataFrame(treename, filename, executor=connection, npartitions=2).Histo1D((col, col, 1, 40, 45), col)
            for col in ["b1", "b2", "b3"]
        ]

        # Before triggering the computation graphs values are None
        for proxy in histoproxies:
            assert proxy.proxied_node.value is None

        ROOT.RDF.RunGraphs(histoproxies)

        # RunGraphs must not return before all computation graphs are done:
        # every value must already be in place at this point, without the need
        # to trigger any further event loop with GetValue()
        for proxy in histoproxies:
            assert proxy.proxied_node.value is not None, (
                "RunGraphs returned before all computation graphs were fully processed"
            )

        # After RunGraphs all histograms are correctly assigned to the
        # node objects
        for proxy in histoproxies:
            histo = proxy.GetValue()
            assert isinstance(histo, ROOT.TH1D)
            assert histo.GetEntries() == nentries
            assert histo.GetMean() == 42


class TestRunGraphsProgressBar:
    """Tests the progressBar option of RunGraphs with the Dask backend"""

    @pytest.mark.parametrize("progressbar", [False, True])
    def test_rungraphs_dask_progressbar(self, payload, progressbar, monkeypatch, capsys):
        """
        Run several independent Dask RDF graphs concurrently with RunGraphs,
        with and without the progress bar, and check that the computations run
        correctly in both cases and that the progress visualization is started
        only when it is requested.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return

        # Imported here because these modules are only available with the Dask backend
        from ROOT._distrdf.Backends.Dask import Backend as DaskBackendModule
        from ROOT._distrdf.TaskProgressBar import TaskProgressBar

        # Spy on the two possible progress-visualization entry points: the
        # DistRDF TaskProgressBar of RunGraphs and the progress function of the
        # Dask backend used for implicitly-triggered computations
        run_calls = []
        original_run = TaskProgressBar.run

        def spy_run(self):
            run_calls.append(self)
            original_run(self)

        progress_calls = []
        monkeypatch.setattr(TaskProgressBar, "run", spy_run)
        monkeypatch.setattr(DaskBackendModule, "progress", lambda *args, **kwargs: progress_calls.append(args))

        treename = "tree"
        filename = "../data/ttree/distrdf_roottest_check_rungraphs.root"
        nentries = 10000
        # Three independent dataframes, i.e. three independent computation graphs
        histoproxies = [
            ROOT.RDataFrame(treename, filename, executor=connection, npartitions=2).Histo1D((col, col, 1, 40, 45), col)
            for col in ["b1", "b2", "b3"]
        ]

        assert ROOT.RDF.RunGraphs(histoproxies, progressBar=progressbar) == 3

        # In both cases the values must be already in place after RunGraphs,
        # without the need to trigger any further event loop with GetValue()
        for proxy in histoproxies:
            assert proxy.proxied_node.value is not None
            histo = proxy.GetValue()
            assert isinstance(histo, ROOT.TH1D)
            assert histo.GetEntries() == nentries
            assert histo.GetMean() == 42

        captured = capsys.readouterr()
        if progressbar:
            # The shared progress bar ran exactly once for all the graphs and
            # something was visualized (either the tqdm bar on stderr or the
            # fallback text progress on stdout, depending on whether tqdm is
            # installed)
            assert len(run_calls) == 1
            assert progress_calls == []
            assert captured.out + captured.err != ""
        else:
            # No progress visualization of any kind was triggered
            assert run_calls == []
            assert progress_calls == []
            assert captured.out == "" and captured.err == ""


if __name__ == "__main__":
    pytest.main(args=[__file__])
