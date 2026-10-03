' ONFR REVO - lanzador sin ventana negra.
' 1. Si REVO ya esta abierto, solo abre la web (al instante).
' 2. Si no, abre al momento una pagina de "Abriendo REVO..." que salta sola
'    a REVO en cuanto esta listo, y arranca REVO en segundo plano.
Option Explicit
Dim sh, fso, dir, url, http, ready
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
url = "http://127.0.0.1:7860/"

ready = False
On Error Resume Next
Set http = CreateObject("MSXML2.ServerXMLHTTP")
http.setTimeouts 500, 500, 500, 500
http.Open "GET", url, False
http.Send
If Err.Number = 0 Then
    If http.Status = 200 Then ready = True
End If
On Error GoTo 0

If ready Then
    sh.Run url
Else
    OpenInBrowser "file:///" & Replace(dir, "\", "/") & "/revo_cargando.html"
    sh.Environment("Process")("REVO_HIDDEN") = "1"
    sh.Run "cmd /c """"" & dir & "\Abrir ONFR REVO.bat""""", 0, False
End If

' Abre una direccion en el navegador predeterminado (el de las paginas web,
' no el programa asociado a los .html, que a veces es el Bloc de notas)
Sub OpenInBrowser(target)
    Dim progId, cmd
    On Error Resume Next
    progId = sh.RegRead("HKCU\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice\ProgId")
    cmd = ""
    If progId <> "" Then cmd = sh.RegRead("HKCR\" & progId & "\shell\open\command\")
    On Error GoTo 0
    If cmd <> "" And InStr(cmd, "%1") > 0 Then
        sh.Run Replace(cmd, "%1", target)
    Else
        sh.Run "explorer.exe """ & target & """"
    End If
End Sub
