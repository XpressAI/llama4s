/**
 * *****************************************************************************
 * Global Settings
 * *****************************************************************************
 */

Global / cancelable := true
Global / onChangedBuildSource := ReloadOnSourceChanges

lazy val defaultScalaVersion = "3.5.1"
ThisBuild / scalaVersion := defaultScalaVersion
ThisBuild / organization := "ai.xpress"

inThisBuild(
  List(
    scalaVersion := defaultScalaVersion,
    semanticdbEnabled := true,
    semanticdbVersion := scalafixSemanticdb.revision
  )
)

lazy val LangChain4JVersion = "0.35.0"

libraryDependencies ++= Seq(
  "org.scala-lang" %% "scala3-library" % defaultScalaVersion,
  "com.lihaoyi" %% "mainargs" % "0.7.6",
  "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4",
  "org.scalatest" %% "scalatest" % "3.2.19" % Test
)

lazy val root = project
  .in(file("."))
  .settings(
    name := "llmtest",
    ThisBuild / version := "0.1.0",
    ThisBuild / scalacOptions += "-Wunused:all"
  )
