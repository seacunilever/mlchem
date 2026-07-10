@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
if "%PYTHON%" == "" (
	set PYTHON=py -3
)
%PYTHON% -V >NUL 2>NUL
if errorlevel 1 (
	python -V >NUL 2>NUL
	if errorlevel 1 (
		echo.
		echo.Neither 'py -3' nor 'python' is available.
		echo.Install Python or set the PYTHON environment variable to a valid interpreter.
		exit /b 1
	)
	set PYTHON=python
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

REM Build HTML and immediately mirror the result into docs\ for GitHub Pages.
if /I "%1" == "html" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
	if errorlevel 1 goto end
	%PYTHON% "%~dp0_publish.py"
	if errorlevel 1 goto end
	goto end
)

REM Standalone mirror step (use after a manual sphinx-build invocation).
if /I "%1" == "publish" (
	%PYTHON% "%~dp0_publish.py"
	if errorlevel 1 goto end
	goto end
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
exit /b %errorlevel%
